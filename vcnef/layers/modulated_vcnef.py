import torch
import torch.nn as nn
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList
import torch.nn.functional as F

from vcnef.linear_transformers.masking import FullMask, LengthMask
from vcnef.linear_transformers.feature_maps.base import elu_feature_map


class VCNeF(Module):
    """
    Vectorized Conditional Neural Field.

    Arguments
    ----------
        layers: list, ModulatedVCNeFLayer instances
        norm_layer: A normalization layer to be applied to the final output
            (default: None which means no normalization)
    """
    def __init__(self, layers, norm_layer=None):
        super(VCNeF, self).__init__()
        self.layers = ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, coordinates):
        # Repeat x for all timesteps
        T = coordinates.shape[1]
        x = x.unsqueeze(1).repeat((1, T, 1, 1))

        # Apply all the vcnef layers
        for layer in self.layers:
            x = layer(x, coordinates)

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x


class VCNeFLayer(Module):
    """
    Modulated VCNeF layer (i.e., one block).

    Arguments
    ---------
        self_attention: The attention implementation to use for self attention
        neural_field: The neural field implementation to use in the VCNeF layer
        d_model: The input feature dimensionality
        dropout: The dropout rate to apply to the intermediate features (default: 0.1)
    """
    def __init__(self, self_attention, neural_field, d_model, dropout=0.1):
        super(VCNeFLayer, self).__init__()
        self.self_attention = self_attention
        self.norm = LayerNorm(d_model)
        self.dropout = Dropout(dropout)

        self.neural_field = neural_field

    def forward(self, x, coordinates):
        """
        Applies the VCNeF to the input x using the coordinates.

        Arguments
        ---------
            x: The input conditioning factor of shape (N, L, D) where N is the batch size,
               L is the sequence length (spatial points) and D should be the same as
               the d_model passed in the constructor.
            coordinates: The coordinates features of shape (N, L, D) where N is the
                batch size, L is the sequence length (coordinates) and
                E should be the same as the d_model.
        """
        # Generate masks for self-attention
        N = x.shape[0]
        L = x.shape[2]
        x_mask = FullMask(L, device=x.device)
        x_length_mask = LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # First apply the self attention and add it to the input
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            query_lengths=x_length_mask,
            key_lengths=x_length_mask
        ))
        x = self.norm(x)

        # Second apply the neural field
        x = self.neural_field(x, coordinates)

        return x


class ModulatedNeuralField(nn.Module):
    """
    Implements a modulated neural field. The neural field is conditioned by modulating
    the hidden representation of the coordinates. It uses multiple layers, residual connections
    and multiple heads to modulate the hidden representation.

    Arguments
    ---------
        d_model: The feature dimensionality of the coordinates and conditioning factor
        n_heads: Number of heads for the modulation
        d_hidden: Hidden representation dimensionality of the neural field
        activation: {'relu', 'gelu'} Which activation to use for the feed
            forward part of the layer (default: relu)
        feature_map: callable, a callable that applies the feature map to the
            last dimension of a tensor (default: elu(x)+1)
    """

    def __init__(self, d_model, n_heads, d_hidden=None, activation="relu", dropout=0.1, feature_map=None):
        super(ModulatedNeuralField, self).__init__()

        # Modulation is done using different heads
        d_heads = d_model // n_heads
        self.n_heads = n_heads

        # Project the input dimension to subspaces
        self.x_projection = Linear(d_model, d_heads * n_heads)
        self.coordinates_projection = Linear(d_model, d_heads * n_heads)
        self.out_projection = Linear(d_heads * n_heads, d_model)

        # Feature map/ non-linearity for the modulation
        self.feature_map = (feature_map(d_heads) if feature_map else elu_feature_map(d_heads))

        d_hidden = d_hidden or d_model * 4
        self.linear1 = Linear(d_model, d_hidden)
        self.linear2 = Linear(d_hidden, d_model)
        self.activation = getattr(F, activation)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x, coordinates):
        # Extract the dimensions into local variables
        N, T, L, _ = x.shape
        H = self.n_heads

        y = x

        # Project the x and coordinates
        x = self.x_projection(x).view(N, T, L, H, -1)
        coordinates = self.coordinates_projection(coordinates).view(N, T, L, H, -1)

        # Compute the modulation
        self.feature_map.new_feature_map(x.device)
        x = (self.feature_map.forward_queries(x) * coordinates).view(N, T, L, -1)

        # Project the output
        x = self.out_projection(x)

        # Apply residual connection
        x = self.norm1(y + self.dropout(x))

        # Apply remaining fully connected layers
        y = x
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y)


class ModulatedNeuralFieldMLP(nn.Module):
    """
    Implements a modulated neural field with more layers. The neural field is conditioned by modulating
    the hidden representation of the coordinates. It uses multiple layers, residual connections
    and multiple heads to modulate the hidden representation.

    Arguments
    ---------
        d_model: The feature dimensionality of the coordinates and conditioning factor
        n_heads: Number of heads for the modulation
        d_hidden: Hidden representation dimensionality of the neural field
        activation: {'relu', 'gelu'} Which activation to use for the feed
            forward part of the layer (default: relu)
        feature_map: callable, a callable that applies the feature map to the
            last dimension of a tensor (default: elu(x)+1)
    """

    def __init__(self, d_model, n_heads, d_hidden=None, activation="relu", dropout=0.1, feature_map=None):
        super(ModulatedNeuralFieldMLP, self).__init__()

        # Modulation is done using different heads
        d_heads = d_model // n_heads
        self.n_heads = n_heads

        # Project the input dimension to subspaces
        self.x_projection = Linear(d_model, d_heads * n_heads)
        self.coordinates_projection = Linear(d_model, d_heads * n_heads)
        self.out_projection = Linear(d_heads * n_heads, d_model)

        # Feature map/ non-linearity for the modulation
        self.feature_map = (feature_map(d_heads) if feature_map else elu_feature_map(d_heads))

        d_hidden = d_hidden or d_model * 4
        self.linear1 = Linear(d_model, d_hidden)
        self.linear2 = Linear(d_hidden, d_model)
        self.activation = getattr(F, activation)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)

        self.coordinates_mlp = torch.nn.Sequential(torch.nn.GELU(),
                                         torch.nn.Linear(d_model, d_model * 2),
                                         torch.nn.GELU(),
                                         torch.nn.Linear(d_model * 2, d_model))

    def forward(self, x, coordinates):
        # Extract the dimensions into local variables
        N, T, L, _ = x.shape
        H = self.n_heads

        y = x

        # Project the x/coordinates
        x = self.x_projection(x).view(N, T, L, H, -1)
        coordinates = self.coordinates_projection(coordinates)

        # Apply MLP
        coordinates = (coordinates + self.coordinates_mlp(coordinates)).view(N, T, L, H, -1)

        # Compute the modulation
        self.feature_map.new_feature_map(x.device)
        x = (self.feature_map.forward_queries(x) * coordinates).view(N, T, L, -1)

        # Project the output
        x = self.out_projection(x)

        # Apply residual connection
        x = self.norm1(y + self.dropout(x))

        # Apply remaining fully connected layers
        y = x
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y)
