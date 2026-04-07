import torch
import torch.nn as nn
import torch.nn.functional as F

from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

from ..norm import RMSNorm
from .masking import check_attention_mask_broadcasts


@BlockRegistry.register()
class MultiQueryAttention(Block):
    """
    Multi-Query Attention.

    This block allows sharing Keys and Values across all Query heads to radically
    reduce memory bandwidth during inference.

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads for the Queries.
        dropout (float): Dropout probability applied to attention weights.
        use_q_norm (bool): If True, applies RMSNorm to queries.
        use_k_norm (bool): If True, applies RMSNorm to keys.
    """

    def __init__(self, emb_dim=256, num_heads=8, dropout=0.0, use_q_norm=False, use_k_norm=False):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        # Query projections: separate for each head (num_heads * head_dim -> emb_dim)
        self.wq = nn.Linear(emb_dim, emb_dim, bias=False)

        # Key/Value projections: single for all heads (emb_dim -> head_dim)
        self.wk = nn.Linear(emb_dim, self.head_dim, bias=False)
        self.wv = nn.Linear(emb_dim, self.head_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim) if use_q_norm else None
        self.k_norm = RMSNorm(self.head_dim) if use_k_norm else None

        # Output projection
        self.wo = nn.Linear(emb_dim, emb_dim, bias=False)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, context=None, value_context=None, *, mask=None, **kwargs):
        """
        Q from ``x``; K from ``context``; V from ``value_context``.
        All default to ``x`` for self-attention.

        Args:
            x (torch.Tensor): Query tensor of shape (batch, tgt_len, emb_dim).
            context (torch.Tensor, optional): Key source. Defaults to x.
            value_context (torch.Tensor, optional): Value source. Defaults to context.
            mask (torch.Tensor, optional): Boolean mask broadcastable to attention scores.
            **kwargs: Ignored.

        Returns:
            torch.Tensor: The output tensor of shape (batch, tgt_len, emb_dim).
        """
        if context is None:
            context = x
        if value_context is None:
            value_context = context

        batch_size = x.size(0)
        tgt_len = x.size(1)

        Q = self.wq(x)
        K = self.wk(context)
        V = self.wv(value_context)

        # Reshape queries to separate heads: (batch, tgt_len, num_heads, head_dim)
        Q = Q.view(batch_size, tgt_len, self.num_heads, self.head_dim)

        # Transpose for batched matmul: (batch, num_heads, tgt_len, head_dim)
        Q = Q.transpose(1, 2)

        # Expand Keys/Values to match the heads dimension for broadcasting:
        # (batch, num_heads, src_len, head_dim)
        K = K.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        V = V.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        if self.q_norm:
            Q = self.q_norm(Q)
        if self.k_norm:
            K = self.k_norm(K)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        if mask is not None:
            check_attention_mask_broadcasts(scores, mask, name="mask")
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # contextualize
        context = torch.matmul(attn_weights, V)  # (batch, num_heads, tgt_len, head_dim)

        # Concatenate heads and project back to emb_dim
        context = context.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.emb_dim)
        output = self.wo(context)

        return output
