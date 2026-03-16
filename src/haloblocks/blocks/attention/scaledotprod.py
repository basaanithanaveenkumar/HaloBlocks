import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

from .utils import RMSNorm

@BlockRegistry.register("scaled_dot_product_attention")
class ScaledDotProductAttention(Block):
    """
    Standard Scaled Dot-Product Attention mechanism.

    This block computes attention as:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Args:
        dropout (float): Dropout probability applied to attention weights.
            Defaults to 0.1.
        head_dim (int, optional): Dimension of the attention heads. Required if q_norm or k_norm is True.
        use_q_norm (bool): If True, applies RMSNorm to queries.
        use_k_norm (bool): If True, applies RMSNorm to keys.
    """
    def __init__(self, dropout=0.1, head_dim=None, use_q_norm=False, use_k_norm=False):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.q_norm = RMSNorm(head_dim) if (use_q_norm and head_dim) else None
        self.k_norm = RMSNorm(head_dim) if (use_k_norm and head_dim) else None

    def forward(self, q, k, v, mask=None, **kwargs):
        """
        Computes the attention output.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            mask (torch.Tensor, optional): Mask to apply to attention scores.
            **kwargs: Ignored.

        Returns:
            torch.Tensor: The attention output.
        """
        if self.q_norm:
            q = self.q_norm(q)
        if self.k_norm:
            k = self.k_norm(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(attn, v)