"""Decoder blocks for HaloBlocks."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base import Block
from .attention import SelfAttentionBlock, CrossAttentionBlock
from .layers import FeedForwardBlock, LayerNormBlock, DropoutBlock


class TransformerDecoderLayerBlock(Block):
    """A single Transformer decoder layer block.

    Consists of a masked self-attention sub-layer, a cross-attention sub-layer,
    and a feed-forward sub-layer, each with residual connection and layer norm.

    Args:
        d_model: Model dimensionality.
        num_heads: Number of attention heads.
        d_ff: Feed-forward inner dimensionality.
        dropout: Dropout probability. Default: 0.1.
        activation: Activation for the feed-forward block ('relu' or 'gelu').
            Default: 'gelu'.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, num_heads, dropout=dropout, causal=True)
        self.cross_attn = CrossAttentionBlock(d_model, num_heads, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, dropout=dropout, activation=activation)
        self.norm1 = LayerNormBlock(d_model)
        self.norm2 = LayerNormBlock(d_model)
        self.norm3 = LayerNormBlock(d_model)
        self.dropout = DropoutBlock(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process one decoder layer.

        Args:
            x: Target tensor of shape (batch, tgt_len, d_model).
            memory: Encoder output of shape (batch, src_len, d_model).
            tgt_key_padding_mask: Optional boolean mask of shape (batch, tgt_len).
            memory_key_padding_mask: Optional boolean mask of shape (batch, src_len).

        Returns:
            Tuple of (output, self_attn_weights, cross_attn_weights) where
            output has shape (batch, tgt_len, d_model).
        """
        sa_out, sa_weights = self.self_attn(x, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + self.dropout(sa_out))

        ca_out, ca_weights = self.cross_attn(x, memory, key_padding_mask=memory_key_padding_mask)
        x = self.norm2(x + self.dropout(ca_out))

        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x, sa_weights, ca_weights


class TransformerDecoderBlock(Block):
    """A stack of Transformer decoder layer blocks.

    Args:
        d_model: Model dimensionality.
        num_heads: Number of attention heads.
        d_ff: Feed-forward inner dimensionality.
        num_layers: Number of decoder layers to stack.
        dropout: Dropout probability. Default: 0.1.
        activation: Activation for the feed-forward block ('relu' or 'gelu').
            Default: 'gelu'.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayerBlock(d_model, num_heads, d_ff, dropout, activation)
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNormBlock(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pass the target through all decoder layers.

        Args:
            x: Target tensor of shape (batch, tgt_len, d_model).
            memory: Encoder output of shape (batch, src_len, d_model).
            tgt_key_padding_mask: Optional boolean mask of shape (batch, tgt_len).
            memory_key_padding_mask: Optional boolean mask of shape (batch, src_len).

        Returns:
            Decoded tensor of shape (batch, tgt_len, d_model).
        """
        for layer in self.layers:
            x, _, _ = layer(
                x,
                memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.norm(x)
