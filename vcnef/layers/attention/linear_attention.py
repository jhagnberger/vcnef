# Code adapted from Linear Transformers (https://github.com/idiap/fast-transformers) of Katharopoulos et al.

import torch
from torch.nn import Module

from vcnef.linear_transformers.feature_maps.base import elu_feature_map


class VectorizedLinearAttention(Module):
    """Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity. This class allows doing N x T attention calculations
    at once and provides a speed-up.

    Given the queries, keys and values as Q, K, V instead of computing

        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),

    we make use of a feature map function Φ(.) and perform the following
    computation

        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).

    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6):
        super(VectorizedLinearAttention, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones
        if not attn_mask.all_ones:
            raise RuntimeError(("LinearAttention does not support arbitrary "
                                "attention masks"))

        K = K * key_lengths.float_matrix[:, None, :, None, None].repeat((1, Q.size(1), 1, 1, 1))

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity

        KV = torch.einsum("ntshd,ntshm->nthmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("ntlhd,nthd->ntlh", Q, K.sum(dim=2))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("ntlhd,nthmd,ntlh->ntlhm", Q, KV, Z)

        return V.contiguous()
