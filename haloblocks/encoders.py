"""Encoder blocks for HaloBlocks."""

from typing import Optional

import torch
import torch.nn as nn

from .base import Block
from .attention import SelfAttentionBlock
from .layers import FeedForwardBlock, LayerNormBlock, DropoutBlock


class TransformerEncoderLayerBlock(Block):
    """A single Transformer encoder layer block.

    Consists of a self-attention sub-layer followed by a feed-forward
    sub-layer, each wrapped with residual connection and layer normalisation.

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
        self.self_attn = SelfAttentionBlock(d_model, num_heads, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, dropout=dropout, activation=activation)
        self.norm1 = LayerNormBlock(d_model)
        self.norm2 = LayerNormBlock(d_model)
        self.dropout = DropoutBlock(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process one encoder layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            src_key_padding_mask: Optional boolean padding mask of shape
                (batch, seq_len).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        attn_out, _ = self.self_attn(x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class TransformerEncoderBlock(Block):
    """A stack of Transformer encoder layer blocks.

    Args:
        d_model: Model dimensionality.
        num_heads: Number of attention heads.
        d_ff: Feed-forward inner dimensionality.
        num_layers: Number of encoder layers to stack.
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
                TransformerEncoderLayerBlock(d_model, num_heads, d_ff, dropout, activation)
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNormBlock(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pass the input through all encoder layers.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            src_key_padding_mask: Optional boolean padding mask of shape
                (batch, seq_len).

        Returns:
            Encoded tensor of shape (batch, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)
