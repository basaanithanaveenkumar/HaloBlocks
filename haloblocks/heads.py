"""Head blocks: task-specific output layers."""

from typing import Optional

import torch
import torch.nn as nn

from .base import Block


class ClassificationHead(Block):
    """Classification head that maps hidden states to class logits.

    Args:
        d_model: Dimensionality of the input hidden states.
        num_classes: Number of output classes.
        dropout: Dropout probability applied before the linear layer. Default: 0.0.
        pooling: Pooling strategy over the sequence dimension.
            'cls' uses the first token, 'mean' averages all tokens. Default: 'cls'.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dropout: float = 0.0,
        pooling: str = "cls",
    ):
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got '{pooling}'.")
        self.pooling = pooling
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(d_model, num_classes)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute class logits.

        Args:
            hidden_states: Tensor of shape (batch, seq_len, d_model).
            attention_mask: Optional boolean mask of shape (batch, seq_len)
                used when pooling='mean' to ignore padding tokens.

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        if self.pooling == "cls":
            pooled = hidden_states[:, 0, :]
        else:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(dim=1)
        return self.linear(self.dropout(pooled))


class LanguageModelHead(Block):
    """Language model head that maps hidden states to vocabulary logits.

    Optionally ties weights with a token embedding matrix.

    Args:
        d_model: Dimensionality of the input hidden states.
        vocab_size: Size of the vocabulary.
        bias: If True, adds bias to the output projection. Default: False.
    """

    def __init__(self, d_model: int, vocab_size: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size, bias=bias)

    def tie_weights(self, embedding_weight: torch.Tensor) -> None:
        """Tie the projection weight to an embedding weight matrix.

        Args:
            embedding_weight: Weight tensor from an Embedding layer.
        """
        self.linear.weight = embedding_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute vocabulary logits.

        Args:
            hidden_states: Tensor of shape (batch, seq_len, d_model).

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
        """
        return self.linear(hidden_states)


class TokenClassificationHead(Block):
    """Token-level classification head for tasks like NER or POS tagging.

    Args:
        d_model: Dimensionality of the input hidden states.
        num_classes: Number of output classes per token.
        dropout: Dropout probability applied before the linear layer. Default: 0.0.
    """

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute per-token class logits.

        Args:
            hidden_states: Tensor of shape (batch, seq_len, d_model).

        Returns:
            Logits tensor of shape (batch, seq_len, num_classes).
        """
        return self.linear(self.dropout(hidden_states))
