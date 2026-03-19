import torch
import torch.nn as nn
import math

from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

@BlockRegistry.register("sinusoidal_positional_embedding")
class SinusoidalPositionalEmbedding(Block):
    """
    Sinusoidal positional embedding (absolute, fixed).

    Adds fixed sinusoidal position encodings to the input token embeddings.
    The encoding is computed once and cached.

    Args:
        emb_dim (int): Dimensionality of the embeddings.
        max_len (int, optional): Maximum sequence length. Defaults to 5000.
        dropout (float, optional): Dropout applied after adding embeddings.
    """
    def __init__(self, emb_dim, max_len=5000, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Create positional encoding matrix (max_len, emb_dim)
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if emb_dim % 2 == 1:
            # Handle odd embedding dimension (last column as cos with same div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, emb_dim)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x, **kwargs):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, emb_dim).

        Returns:
            torch.Tensor: x + positional encodings (with dropout).
        """
        seq_len = x.size(1)
        # Add positional encoding (truncate to seq_len)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
