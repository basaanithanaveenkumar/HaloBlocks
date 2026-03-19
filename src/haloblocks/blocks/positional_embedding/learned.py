import torch
import torch.nn as nn

from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

@BlockRegistry.register("learned_positional_embedding")
class LearnedPositionalEmbedding(Block):
    """
    Learned absolute positional embedding.

    Each position up to `max_len` gets a learnable embedding vector.

    Args:
        emb_dim (int): Dimensionality of the embeddings.
        max_len (int): Maximum sequence length.
        dropout (float, optional): Dropout applied after adding embeddings.
    """
    def __init__(self, emb_dim, max_len, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Learnable embedding table
        self.pe = nn.Embedding(max_len, emb_dim)
        # Initialize with normal distribution (optional)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x, **kwargs):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, emb_dim).

        Returns:
            torch.Tensor: x + positional embeddings (with dropout).
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_len}")
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        pos_emb = self.pe(positions)  # (1, seq_len, emb_dim)
        x = x + pos_emb
        return self.dropout(x)
