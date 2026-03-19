import torch

from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

@BlockRegistry.register("rotary_positional_embedding")
class RotaryPositionalEmbedding(Block):
    """
    Rotary Positional Embedding (RoPE).

    Applies rotation to queries and keys based on their absolute positions.
    This block is designed to be called inside attention before computing scores.

    Args:
        head_dim (int): Dimension of each attention head.
        max_len (int, optional): Maximum sequence length for precomputed frequencies.
        base (float, optional): Base for the geometric progression of frequencies.
    """
    def __init__(self, head_dim, max_len=2048, base=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_len = max_len
        self.base = base

        # Precompute frequency inverses
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute sinusoidal cache for positions up to max_len
        self._build_cache(max_len)

    def _build_cache(self, max_len):
        # positions: (max_len,)
        positions = torch.arange(max_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        # frequencies: (max_len, head_dim//2)
        freqs = torch.einsum('i,j->ij', positions, self.inv_freq)  # (max_len, head_dim//2)
        # emb: (max_len, head_dim)
        emb = torch.cat((freqs, freqs), dim=-1)  # duplicate for even/odd indices
        # cos and sin: (max_len, head_dim)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len=None, **kwargs):
        """
        Apply rotary embedding to queries and keys.

        Args:
            q (torch.Tensor): Query of shape (batch, num_heads, tgt_len, head_dim)
            k (torch.Tensor): Key of shape (batch, num_heads, src_len, head_dim)
            seq_len (int, optional): If provided, uses only first `seq_len` positions.
                Otherwise, uses the maximum of q_len and k_len.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated q and k.
        """
        # Determine sequence length for cache
        q_len = q.size(2)
        k_len = k.size(2)
        if seq_len is None:
            seq_len = max(q_len, k_len)
        if seq_len > self.max_len:
            # Dynamically extend cache (or recompute on the fly)
            # For simplicity, we rebuild cache if needed (could be optimized)
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len]  # (seq_len, head_dim)
        sin = self.sin_cached[:seq_len]

        # Apply rotation
        # q: (batch, heads, tgt_len, head_dim) -> apply to last dimension
        q_rot = (q * cos[:q_len]) + (self._rotate_half(q) * sin[:q_len])
        k_rot = (k * cos[:k_len]) + (self._rotate_half(k) * sin[:k_len])
        return q_rot, k_rot
