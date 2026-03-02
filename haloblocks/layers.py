"""Layer blocks: fundamental building blocks for neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Block


class LinearBlock(Block):
    """A linear (fully-connected) layer block with optional bias.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias. Default: True.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LayerNormBlock(Block):
    """Layer normalisation block.

    Args:
        normalized_shape: Input shape from an expected input of a given size.
        eps: Value added to denominator for numerical stability. Default: 1e-5.
    """

    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class DropoutBlock(Block):
    """Dropout block.

    Args:
        p: Probability of an element to be zeroed. Default: 0.1.
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


class FeedForwardBlock(Block):
    """Position-wise feed-forward block used in Transformer models.

    Consists of two linear layers with a GELU activation and dropout.

    Args:
        d_model: Model (input/output) dimensionality.
        d_ff: Inner dimensionality of the feed-forward layer.
        dropout: Dropout probability. Default: 0.1.
        activation: Activation function ('relu' or 'gelu'). Default: 'gelu'.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation '{activation}'. Use 'gelu' or 'relu'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class EmbeddingBlock(Block):
    """Token embedding block.

    Args:
        vocab_size: Size of the vocabulary.
        d_model: Embedding dimensionality.
        padding_idx: If given, pads the output with zeros whenever the input
            index equals this value. Default: None.
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)
