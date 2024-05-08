import torch
import torch.nn as nn
from torch.nn import LayerNorm

# Import Linear Transformers
from vcnef.linear_transformers.attention.linear_attention import LinearAttention
from vcnef.linear_transformers.attention.attention_layer import AttentionLayer
from vcnef.linear_transformers.transformers import TransformerEncoderLayer, TransformerEncoder

# Import VCNeF components
from vcnef.layers.attention.linear_attention import VectorizedLinearAttention
from vcnef.layers.attention.attention_layer import VectorizedAttentionLayer
from vcnef.layers.modulated_vcnef import VCNeF, VCNeFLayer, ModulatedNeuralField


class VCNeFModel(nn.Module):
    """
    VCNeF for 1D time-dependent PDEs.

    Vectorized Contitional Neural Field with linear self-attention for solving PDEs.
    The VCNeF model directly predicts the solution of the PDE of a queried time t. The
    prediction of multiple timesteps of a trajectory is calculated in parallel.

    Arguments
    ---------
        num_channels: Number of input channels of the PDE
        condition_on_pde_param: True, if the model is also conditioned on the PDE parameter value.
        pde_param_dim: Dimension of the PDE parameter value (e.g., 1 for viscosity parameter of 1D Burgers')
        d_model: Dimensionality of the hidden representation for each spatial point
        n_heads: Number of heads for the Transfomer block(s)
        n_transformer_blocks: Number of Transformer blocks
        n_modulation_blocks: Number of modulation/ VCNeF blocks
    """

    def __init__(self, num_channels=1, condition_on_pde_param=True, pde_param_dim=1,
                 d_model=96, n_heads=8, n_transformer_blocks=3, n_modulation_blocks=3):
        super(VCNeFModel, self).__init__()

        self.use_pde_param = condition_on_pde_param
        input_dim = 1 + num_channels + pde_param_dim if condition_on_pde_param else 1 + num_channels
        d_query = d_model // n_heads

        self.initial_value_encoder = nn.Linear(input_dim, d_model)
        self.coordinates_encoder = nn.Linear(2, d_model)

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
                    "relu"
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
                    ModulatedNeuralField(
                        d_model,
                        n_heads,
                        activation="relu"
                    ),
                    d_model,
                    0.1
                    )
                for _ in range(n_modulation_blocks)
            ],
            LayerNorm(d_model))

        self.decoder = nn.Sequential(nn.Linear(d_model, d_model),
                                     nn.GELU(),
                                     nn.Linear(d_model, num_channels))

    def forward(self, x, grid, pde_param, t):
        # x shape: b, s_x, channels

        # Add grid as positional encoding and concatenate PDE parameter value to initial condition
        x = torch.cat((grid, x), dim=-1)
        if self.use_pde_param:
            pde_param = pde_param[:, None, :].repeat(1, x.size(1), 1)
            x = torch.cat((x, pde_param), dim=-1)

        # Encode and apply transformer
        x = self.initial_value_encoder(x)
        x = self.transformer(x)

        # Repeat spatio-temporal coordiantes
        t = (t[:, :, None, None]).repeat((1, 1, x.size(1), 1))
        grid = (grid[:, None, :, :]).repeat((1, t.size(1), 1, 1))
        spatio_temporal_coordinates = torch.cat((t, grid), dim=-1)

        # Generate latent representation of spatio-temporal coordinates
        spatio_temporal_coordinates = self.coordinates_encoder(spatio_temporal_coordinates)

        # Apply VCNeF blocks/ modulation blocks
        x = self.vcnef(x, spatio_temporal_coordinates)

        # Decode and permute dimensions
        x = self.decoder(x)
        x = x.permute((0, 2, 1, 3))

        # x shape: b, s_x, n_t, channels
        return x
