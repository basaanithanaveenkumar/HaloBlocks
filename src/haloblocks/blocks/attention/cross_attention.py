import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

from ..norm import RMSNorm
from .masking import check_attention_mask_broadcasts


@BlockRegistry.register()
class CrossAttention(Block):
    """
    Single-head cross-attention: keys are projected from ``context``; values use
    ``value_context`` when provided, otherwise the same tensor as keys.

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        return_attn_weights (bool): If True, returns both the output and the
            attention weights. Defaults to False.
        use_q_norm (bool): If True, applies QK-Norm (RMSNorm) to queries.
        use_k_norm (bool): If True, applies QK-Norm (RMSNorm) to keys.
    """

    def __init__(self, emb_dim, return_attn_weights=False, use_q_norm=False, use_k_norm=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.return_attn_weights = return_attn_weights

        # Linear projections
        self.query_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.val_proj = nn.Linear(emb_dim, emb_dim, bias=False)

        self.q_norm = RMSNorm(emb_dim) if use_q_norm else None
        self.k_norm = RMSNorm(emb_dim) if use_k_norm else None

        self.scale = 1.0 / math.sqrt(emb_dim)

    def forward(self, x, context, value_context=None, *, mask=None, **kwargs):
        """Q from ``x``; K from ``context``; V from ``value_context`` when given, else ``context``."""
        val_src = context if value_context is None else value_context
        Query = self.query_proj(x)
        Key = self.key_proj(context)
        Val = self.val_proj(val_src)

        if self.q_norm:
            Query = self.q_norm(Query)
        if self.k_norm:
            Key = self.k_norm(Key)

        attn_scores = torch.matmul(Query, Key.transpose(-2, -1)) * self.scale

        if mask is not None:
            check_attention_mask_broadcasts(attn_scores, mask, name="mask")
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, Val)

        if self.return_attn_weights:
            return attn_output, attn_weights
        else:
            return attn_output


@BlockRegistry.register()
class HeadCrossAttention(Block):
    """
    A single cross-attention head.

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        head_size (int): Dimensionality of the head (typically emb_dim // num_heads).
        drop_fact (float): Dropout probability. Defaults to 0.0.
        causal_mask (bool): If True, applies a causal mask for auto-regressive decoding.
        return_attn_weights (bool): If True, returns both the output and the attention weights.
    """

    def __init__(
        self,
        emb_dim=256,
        head_size=16,
        drop_fact=0.0,
        causal_mask=False,
        return_attn_weights=False,
        use_q_norm=False,
        use_k_norm=False,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.head_size = head_size
        self.causal_mask = causal_mask
        self.return_attn_weights = return_attn_weights

        self.query_proj = nn.Linear(emb_dim, head_size, bias=False)
        self.key_proj = nn.Linear(emb_dim, head_size, bias=False)
        self.val_proj = nn.Linear(emb_dim, head_size, bias=False)

        self.q_norm = RMSNorm(head_size) if use_q_norm else None
        self.k_norm = RMSNorm(head_size) if use_k_norm else None

        self.scale = 1.0 / math.sqrt(head_size)
        self.dropout = nn.Dropout(drop_fact)

    def forward(self, x, context, value_context=None, **kwargs):
        """Q from ``x``; K from ``context``; V from ``value_context`` or ``context``."""
        val_src = context if value_context is None else value_context
        Query = self.query_proj(x)
        Key = self.key_proj(context)
        Val = self.val_proj(val_src)

        if self.q_norm:
            Query = self.q_norm(Query)
        if self.k_norm:
            Key = self.k_norm(Key)

        scores = torch.matmul(Query, Key.transpose(-2, -1)) * self.scale

        if self.causal_mask:
            seq_len_q = x.shape[1]
            seq_len_kv = context.shape[1]
            causal_tril = torch.tril(torch.ones(seq_len_q, seq_len_kv, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(causal_tril == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, Val)

        if self.return_attn_weights:
            return attn_output, attn_weights
        else:
            return attn_output


@BlockRegistry.register()
class MultiHeadCrossAttention(Block):
    """
    Multi-Head Cross Attention (MHCA) block.

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        drop_fact (float): Dropout probability. Defaults to 0.0.
        causal_mask (bool): If True, applies causal masking to all heads.
        return_attn_weights (bool): If True, returns both the output and the attention weights.
        use_q_norm (bool): If True, applies QK-Norm (RMSNorm).
        use_k_norm (bool): If True, applies QK-Norm (RMSNorm).
    """

    def __init__(
        self,
        emb_dim=256,
        num_heads=8,
        drop_fact=0.0,
        causal_mask=False,
        return_attn_weights=False,
        use_q_norm=False,
        use_k_norm=False,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")

        self.head_size = emb_dim // num_heads
        self.heads = nn.ModuleList(
            [
                HeadCrossAttention(
                    emb_dim,
                    head_size=self.head_size,
                    drop_fact=drop_fact,
                    causal_mask=causal_mask,
                    return_attn_weights=return_attn_weights,
                    use_q_norm=use_q_norm,
                    use_k_norm=use_k_norm,
                )
                for _ in range(num_heads)
            ]
        )

        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(drop_fact)

    def forward(self, x, context, value_context=None, **kwargs):
        """Q from ``x``; K from ``context``; V from ``value_context`` or ``context``."""
        head_outputs = [head(x, context, value_context) for head in self.heads]

        # Concatenate head outputs
        concat_output = torch.cat(head_outputs, dim=-1)

        # Final linear projection
        output = self.proj(concat_output)
        output = self.dropout(output)

        return output
