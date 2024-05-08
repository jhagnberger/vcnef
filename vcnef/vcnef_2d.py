import math

import torch
import torch.nn as nn
from torch.nn import LayerNorm
from einops import rearrange

# Import Linear Transformers
from vcnef.linear_transformers.attention.linear_attention import LinearAttention
from vcnef.linear_transformers.attention.attention_layer import AttentionLayer
from vcnef.linear_transformers.transformers import TransformerEncoderLayer, TransformerEncoder

# Import VCNeF components
from vcnef.layers.attention.linear_attention import VectorizedLinearAttention
from vcnef.layers.attention.attention_layer import VectorizedAttentionLayer
from vcnef.layers.modulated_vcnef import VCNeF, VCNeFLayer, ModulatedNeuralFieldMLP


class PositionalEncoding(torch.nn.Module):
    """
    Positional encoding for Transformers proposed by Vaswani et al.
    """
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, t):
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(t.device)
        pe = torch.zeros(size=(t.shape[0], t.shape[1], t.shape[2], self.d_model), device=t.device)
        position = t * self.max_len
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)

        return self.dropout(pe)


class PatchEncoding(nn.Module):
    """
    Image to patches of size embed dim
    """
    def __init__(self, patch_size=2, in_chans=6, embed_dim=96):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_chans, embed_dim // 2, kernel_size=patch_size // 2, stride=patch_size // 2, bias=True),
                                  nn.GELU(),
                                  nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=2, stride=2, bias=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class PatchDecoding(nn.Module):
    """
    Patches of size embed dim to image
    """
    def __init__(self, patch_size=2, out_chans=3, embed_dim=96):
        super().__init__()
        self.conv = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=patch_size // 2, stride=patch_size // 2, bias=True),
                                  nn.GELU(),
                                  nn.ConvTranspose2d(embed_dim // 2, out_chans, kernel_size=2, stride=2, bias=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, G: int, M: int, F_dim: int, H_dim: int, D: int, gamma: float):
        """
        Learnable Fourier Features from https://arxiv.org/pdf/2106.02795.pdf (Algorithm 1)
        Implementation of Algorithm 1: Compute the Fourier feature positional encoding of a multi-dimensional position
        Computes the positional encoding of a tensor of shape [N, G, M]
        :param M: each point has a M-dimensional positional values
        :param F_dim: depth of the Fourier feature dimension
        :param H_dim: hidden layer dimension
        :param D: positional encoding dimension
        :param gamma: parameter to initialize Wr
        """
        super().__init__()
        self.G = G
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = D
        self.gamma = gamma
        self.f_dim_sqrt = math.sqrt(self.F_dim)

        # Projection matrix on learned lines (used in eq. 2)
        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D // self.G)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        """
        Produce positional encodings from x
        :param x: tensor of shape [N, M] that represents N positions where each position is in the shape of [G, M],
                  where G is the positional group and each group has M-dimensional positional values.
                  Positions in different positional groups are independent
        :return: positional encoding for X
        """
        # Step 1. Compute Fourier features (eq. 2)
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = 1 / self.f_dim_sqrt * torch.cat([cosines, sines], dim=-1)
        # Step 2. Compute projected Fourier features (eq. 6)
        Y = self.mlp(F)
        # Step 3. Reshape to x's shape
        PEx = torch.flatten(Y, start_dim=-2)
        return PEx


class VCNeFModel(nn.Module):
    """
    VCNeF for 2D time-dependent PDEs. The input spatial domain is divided into non-overlapping
    patches of two different sizes (multi-scale patching) to allow the model better capturing 
    global and local dynamics.

    Vectorized Contitional Neural Field with linear self-attention for solving PDEs.
    The VCNeF model directly predicts the solution of the PDE of a queried time t. The
    prediction of multiple timesteps of a trajectory is calculated in parallel.

    Arguments
    ---------
        num_channels: Number of input channels of the PDE
        condition_on_pde_param: True, if the model is also conditioned on the PDE parameter value.
        pde_param_dim: Dimension of the PDE parameter value (e.g., 2 for viscosity parameters of 2D Navier-Stokes)
        d_model: Dimensionality of the hidden representation for each spatial point
        n_heads: Number of heads for the Transfomer block(s)
        patch_size_small: Size of small patches (default: 4x4)
        patch_size_large: Size of large patches (default: 16x16)
        n_transformer_blocks: Number of Transformer blocks
        n_modulation_blocks: Number of modulation/ VCNeF blocks
    """

    def __init__(self, num_channels=4, condition_on_pde_param=True, pde_param_dim=2,
                 d_model=256, n_heads=8, patch_size_small=4, patch_size_large=16,
                 n_transformer_blocks=1, n_modulation_blocks=6):
        super(VCNeFModel, self).__init__()

        self.condition_on_pde_param = condition_on_pde_param
        self.patch_size_small = patch_size_small
        self.patch_size_large = patch_size_large
        input_dim = 2 + num_channels + pde_param_dim if condition_on_pde_param else 2 + num_channels
        d_query = d_model // n_heads

        self.patch_encoding_small = PatchEncoding(in_chans=input_dim, embed_dim=d_model, patch_size=patch_size_small)
        self.patch_encoding_large = PatchEncoding(in_chans=input_dim, embed_dim=d_model, patch_size=patch_size_large)

        self.time_encoding = PositionalEncoding(d_model // 2)
        self.coordinate_encoding_small = LearnableFourierPositionalEncoding(2, 2, d_model // 2, 32, d_model // 2, 10)
        self.coordinate_encoding_large = LearnableFourierPositionalEncoding(2, 2, d_model // 2, 32, d_model // 2, 10)

        self.transformer = TransformerEncoder(
            [
                TransformerEncoderLayer(
                    AttentionLayer(
                        LinearAttention(d_query),
                        d_model,
                        n_heads,
                        d_keys=d_query,
                        d_values=d_query
                    ),
                    d_model,
                    d_model * 4,
                    0.1,
                    "gelu"
                    )
                for _ in range(n_transformer_blocks)
            ],
            LayerNorm(d_model))

        self.vcnef = VCNeF(
            [
                VCNeFLayer(
                    VectorizedAttentionLayer(
                        VectorizedLinearAttention(d_query),
                        d_model,
                        n_heads,
                        d_keys=d_query,
                        d_values=d_query
                    ),
                    ModulatedNeuralFieldMLP(
                        d_model,
                        n_heads,
                        activation="gelu"
                    ),
                    d_model,
                    0.1
                    )
                for _ in range(n_modulation_blocks)
            ],
            LayerNorm(d_model))

        self.final_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model))

        self.patch_decoding_small = PatchDecoding(out_chans=num_channels, embed_dim=d_model, patch_size=patch_size_small)
        self.patch_decoder_large = PatchDecoding(out_chans=num_channels, embed_dim=d_model, patch_size=patch_size_large)
        self.final_conv = nn.Conv2d(num_channels * 2, num_channels, 1, 1)

    def forward(self, x, grid, pde_param, t):
        # x shape: b, s_x, s_y, channels

        # Add grid as positional encoding and concatenate PDE parameter value to initial condition
        x = torch.cat((grid, x), dim=-1)
        if self.condition_on_pde_param:
            pde_param = pde_param[:, None, None, :].repeat(1, x.size(1), x.size(2), 1)
            x = torch.cat((x, pde_param), dim=-1)

        # Generate small and large patches
        x = rearrange(x, 'b h w c -> b c h w')
        x_small = self.patch_encoding_small(x)
        x_large = self.patch_encoding_large(x)

        h_small = x_small.shape[2]
        w_small = x_small.shape[3]
        h_large = x_large.shape[2]
        w_large = x_large.shape[3]

        x_small = rearrange(x_small, 'b c h w -> b (h w) c')
        x_large = rearrange(x_large, 'b c h w -> b (h w) c')
        x = torch.concat((x_small, x_large), dim=1)

        # Apply transformer to patches
        x = self.transformer(x)

        # Encode temporal coordinates
        t = (t[:, :, None, None]).repeat((1, 1, x.size(1), 1))
        temporal_coordinates = self.time_encoding(t)

        # Encode spatial coordinates of small patches
        points_small = torch.cat((grid[:, ::self.patch_size_small, ::self.patch_size_small, None, :],
                                  grid[:, (self.patch_size_small-1)::self.patch_size_small, (self.patch_size_small-1)::self.patch_size_small, None, :]), dim=-2)
        spatial_coordinates_small = self.coordinate_encoding_small(rearrange(points_small, 'b h w g c -> b (h w) g c'))
        spatial_coordinates_small = (spatial_coordinates_small[:, None, :, :]).repeat((1, t.size(1), 1, 1))

        # Encode spatial coordinates of large patches
        points_large = torch.cat((grid[:, ::self.patch_size_large, ::self.patch_size_large, None, :],
                                  grid[:, (self.patch_size_large-1)::self.patch_size_large, (self.patch_size_large-1)::self.patch_size_large, None, :]), dim=-2)
        spatial_coordinates_large = self.coordinate_encoding_large(rearrange(points_large, 'b h w g c -> b (h w) g c'))
        spatial_coordinates_large = (spatial_coordinates_large[:, None, :, :]).repeat((1, t.size(1), 1, 1))

        spatial_coordinates = torch.concat((spatial_coordinates_small, spatial_coordinates_large), dim=-2)

        # Concatenate temporal and spatial coordinates togehter to spatio-temporal coordinates
        spatio_temporal_coordinates = torch.cat([temporal_coordinates, spatial_coordinates], dim=-1)

        # Apply VCNeF blocks/ modulation blocks and a final MLP
        x = self.vcnef(x, spatio_temporal_coordinates)
        x = self.final_mlp(x)

        # Decode small patches
        x_small = x[:, :, :h_small*w_small, :]
        x_small = rearrange(x_small, 'b t (h w) c -> (b t) c h w', h=h_small, w=w_small)
        x_small = self.patch_decoding_small(x_small)

        # Decode large patches
        x_large = x[:, :, h_small*w_small:, :]
        x_large  = rearrange(x_large , 'b t (h w) c -> (b t) c h w', h=h_large, w=w_large)
        x_large  = self.patch_decoder_large(x_large )

        # Combine small and large patches as weighted sum
        x = torch.cat([x_small, x_large], dim=1)
        x = self.final_conv(x)

        # Reshape
        x = rearrange(x, '(b t) c h w -> b h w t c', t=t.shape[1])

        # x shape: b, s_x, s_y, n_t, channels
        return x
