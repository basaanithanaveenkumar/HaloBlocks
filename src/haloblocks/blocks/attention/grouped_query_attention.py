import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

from ..norm import RMSNorm
from .masking import check_attention_mask_broadcasts


@BlockRegistry.register()
class GroupedQueryAttention(Block):
    """
    Grouped-Query Attention (GQA) module.

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of query heads.
        num_kv_heads (int): Number of key/value heads (must divide num_heads evenly).
        dropout (float): Dropout probability.
        use_q_norm (bool): If True, applies RMSNorm to queries.
        use_k_norm (bool): If True, applies RMSNorm to keys.
    """

    def __init__(self, emb_dim=256, num_heads=8, num_kv_heads=2, dropout=0.0, use_q_norm=False, use_k_norm=False):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = emb_dim // num_heads
        self.group_size = num_heads // num_kv_heads

        # Query projection
        self.wq = nn.Linear(emb_dim, emb_dim, bias=False)
        # Key/Value projections
        self.wk = nn.Linear(emb_dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(emb_dim, num_kv_heads * self.head_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim) if use_q_norm else None
        self.k_norm = RMSNorm(self.head_dim) if use_k_norm else None

        # Output projection
        self.wo = nn.Linear(emb_dim, emb_dim, bias=False)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, context=None, *, mask=None, **kwargs):
        """
        Q from ``x``; K and V both projected from ``context`` (defaults to ``x`` for self-attention).

        Args:
            x (torch.Tensor): Input/Query tensor of shape (batch, seq_len_q, emb_dim).
            context (torch.Tensor, optional): Context tensor for keys/values. Defaults to x.
            mask (torch.Tensor, optional): Boolean mask broadcastable to scores
                ``(batch, num_heads, seq_len_q, seq_len_kv)``.
            **kwargs: Ignored.

        Returns:
            torch.Tensor: The output tensor.
        """
        if context is None:
            context = x

        batch_size, seq_len_q, _ = x.shape
        _, seq_len_kv, _ = context.shape

        # Project
        query = self.wq(x)
        key = self.wk(context)
        value = self.wv(context)

        # Reshape to (batch, heads, seq_len, head_dim)
        query = query.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len_kv, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len_kv, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm:
            query = self.q_norm(query)
        if self.k_norm:
            key = self.k_norm(key)

        # Expand K and V to match num_heads
        key = key.repeat_interleave(self.group_size, dim=1)
        value = value.repeat_interleave(self.group_size, dim=1)

        # Compute scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            check_attention_mask_broadcasts(scores, mask, name="mask")
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax and attention output
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        context_output = torch.matmul(attn_weights, value)  # (batch, num_heads, seq_len_q, head_dim)

        # Concatenate and project back
        context_output = context_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.emb_dim)
        output = self.wo(context_output)

        return output
