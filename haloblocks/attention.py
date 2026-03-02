"""Attention blocks for HaloBlocks."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Block


class MultiHeadAttentionBlock(Block):
    """Multi-head attention block.

    Computes multi-head scaled dot-product attention as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    Args:
        d_model: Total dimension of the model.
        num_heads: Number of parallel attention heads.
        dropout: Dropout probability on attention weights. Default: 0.0.
        bias: If True, adds bias to query/key/value projections. Default: True.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale_factor = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split last dimension into (num_heads, head_dim) and transpose."""
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (batch, heads, seq, head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge heads back into d_model dimension."""
        batch, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head attention.

        Args:
            query: Query tensor of shape (batch, tgt_len, d_model).
            key: Key tensor of shape (batch, src_len, d_model).
            value: Value tensor of shape (batch, src_len, d_model).
            attn_mask: Optional additive attention mask of shape
                (tgt_len, src_len) or (batch * num_heads, tgt_len, src_len).
            key_padding_mask: Optional boolean mask of shape (batch, src_len).
                Positions set to True are ignored.

        Returns:
            Tuple of (output, attention_weights) where output has shape
            (batch, tgt_len, d_model) and attention_weights has shape
            (batch, num_heads, tgt_len, src_len).
        """
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale_factor

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        if key_padding_mask is not None:
            # (batch, 1, 1, src_len)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        output = self.out_proj(self._merge_heads(context))
        return output, attn_weights


class SelfAttentionBlock(Block):
    """Self-attention block where query, key, and value all come from the same input.

    Wraps MultiHeadAttentionBlock for convenience.

    Args:
        d_model: Total dimension of the model.
        num_heads: Number of parallel attention heads.
        dropout: Dropout probability on attention weights. Default: 0.0.
        causal: If True, apply a causal (autoregressive) mask. Default: False.
        bias: If True, adds bias to projections. Default: True.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.attn = MultiHeadAttentionBlock(d_model, num_heads, dropout, bias)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build an additive causal mask."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            key_padding_mask: Optional boolean mask of shape (batch, seq_len).

        Returns:
            Tuple of (output, attention_weights).
        """
        attn_mask = None
        if self.causal:
            attn_mask = self._causal_mask(x.size(1), x.device)
        return self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)


class CrossAttentionBlock(Block):
    """Cross-attention block where queries come from one sequence and keys/values from another.

    Wraps MultiHeadAttentionBlock for convenience.

    Args:
        d_model: Total dimension of the model.
        num_heads: Number of parallel attention heads.
        dropout: Dropout probability on attention weights. Default: 0.0.
        bias: If True, adds bias to projections. Default: True.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.attn = MultiHeadAttentionBlock(d_model, num_heads, dropout, bias)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cross-attention.

        Args:
            query: Query tensor of shape (batch, tgt_len, d_model).
            context: Context tensor of shape (batch, src_len, d_model).
            key_padding_mask: Optional boolean mask of shape (batch, src_len).

        Returns:
            Tuple of (output, attention_weights).
        """
        return self.attn(query, context, context, key_padding_mask=key_padding_mask)
