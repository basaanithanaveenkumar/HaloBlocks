import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

@BlockRegistry.register("scaled_dot_product_attention")
class ScaledDotProductAttention(Block):
    """
    Standard Scaled Dot-Product Attention mechanism.

    This block computes attention as:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Args:
        dropout (float): Dropout probability applied to attention weights.
            Defaults to 0.1.
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

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
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(attn, v)