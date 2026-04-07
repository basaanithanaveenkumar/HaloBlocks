import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

from ..norm import RMSNorm


@BlockRegistry.register()
class SlidingWindowAttention(Block):
    """
    Sliding Window Attention mechanism with local attention patterns.

    This attention mechanism restricts attention to a local window around each position,
    reducing computational complexity from O(n²) to O(n * window_size).

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        window_size (int): Size of the sliding window. Defaults to 128.
        use_q_norm (bool): If True, applies RMSNorm to queries. Defaults to True.
        use_k_norm (bool): If True, applies RMSNorm to keys. Defaults to True.
        causal (bool): If True, applies causal masking within the window. Defaults to False.
    """

    def __init__(self, emb_dim=256, num_heads=4, window_size=128, use_q_norm=True, use_k_norm=True, causal=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.causal = causal

        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")

        self.head_dim = emb_dim // num_heads

        # Attention projections
        self.wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wv = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wo = nn.Linear(emb_dim, emb_dim, bias=False)

        # Optional normalization
        self.q_norm = RMSNorm(self.head_dim) if use_q_norm else None
        self.k_norm = RMSNorm(self.head_dim) if use_k_norm else None

    def _create_sliding_window_mask(self, seq_len, device):
        """
        Creates a sliding window attention mask.

        Args:
            seq_len (int): Sequence length.
            device: Torch device.

        Returns:
            torch.Tensor: Mask of shape (1, 1, seq_len, seq_len) where 1 indicates
                         positions that can attend, 0 indicates masked positions.
        """
        # Create indices matrix
        indices = torch.arange(seq_len, device=device)

        # Create mask where each position can attend to positions within window_size
        # For position i, it can attend to positions in [i - window_size + 1, i]
        # if causal, or [i - window_size + 1, i + window_size - 1] if not causal
        if self.causal:
            # Causal sliding window: each position attends to itself and previous window_size-1 tokens
            mask = (indices.unsqueeze(0) - indices.unsqueeze(1) < self.window_size) & (
                indices.unsqueeze(0) - indices.unsqueeze(1) >= 0
            )
        else:
            # Bidirectional sliding window: each position attends to window_size tokens on each side
            distance = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
            mask = distance < self.window_size

        # Add batch and head dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)

        return mask

    def forward(self, x, *, mask=None, **kwargs):
        """
        Computes sliding window attention output.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim).
            mask (torch.Tensor, optional): Additional attention mask to combine with
                                          sliding window mask.

        Returns:
            torch.Tensor: The attention output.
        """
        batch, seq_len, emb_dim = x.shape
        h = self.num_heads
        head_dim = self.head_dim

        # Linear projections and reshape for multi-head attention
        query = self.wq(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)
        key = self.wk(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)
        value = self.wv(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)

        # Apply optional normalization
        if self.q_norm:
            query = self.q_norm(query)
        if self.k_norm:
            key = self.k_norm(key)

        # Create sliding window mask
        window_mask = self._create_sliding_window_mask(seq_len, x.device)

        # Combine with provided mask if any
        if mask is not None:
            # Ensure mask has same shape as window_mask
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            combined_mask = window_mask & mask
        else:
            combined_mask = window_mask

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply mask (set masked positions to large negative value)
        scores = scores.masked_fill(combined_mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)

        # Reshape back to original dimensions
        attention_output = attention_output.transpose(1, 2).reshape(batch, seq_len, emb_dim)

        # Final output projection
        output = self.wo(attention_output)

        return output


@BlockRegistry.register()
class DilatedSlidingWindowAttention(Block):
    """
    Dilated Sliding Window Attention with dilation to increase receptive field.

    This attention mechanism uses dilation to allow each position to attend to
    positions with gaps, increasing the effective receptive field without
    increasing the window size.

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        window_size (int): Size of the sliding window. Defaults to 64.
        dilation (int): Dilation rate. Defaults to 1.
        use_q_norm (bool): If True, applies RMSNorm to queries. Defaults to True.
        use_k_norm (bool): If True, applies RMSNorm to keys. Defaults to True.
        causal (bool): If True, applies causal masking within the window. Defaults to False.
    """

    def __init__(
        self, emb_dim=256, num_heads=4, window_size=64, dilation=1, use_q_norm=True, use_k_norm=True, causal=False
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.dilation = dilation
        self.causal = causal

        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")

        self.head_dim = emb_dim // num_heads

        # Attention projections
        self.wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wv = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wo = nn.Linear(emb_dim, emb_dim, bias=False)

        # Optional normalization
        self.q_norm = RMSNorm(self.head_dim) if use_q_norm else None
        self.k_norm = RMSNorm(self.head_dim) if use_k_norm else None

    def _create_dilated_window_mask(self, seq_len, device):
        """
        Creates a dilated sliding window attention mask.

        Args:
            seq_len (int): Sequence length.
            device: Torch device.

        Returns:
            torch.Tensor: Mask of shape (1, 1, seq_len, seq_len).
        """
        # Create dilated mask
        # For position i, it can attend to positions i - k*dilation where k ranges
        # from 0 to window_size-1 (and also i + k*dilation if not causal)
        if self.causal:
            # Causal dilated window: attend to positions i - k*dilation
            mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
            for i in range(seq_len):
                # Calculate positions within dilated window
                positions = i - torch.arange(0, self.window_size) * self.dilation
                positions = positions[positions >= 0]
                mask[i, positions] = 1
        else:
            # Bidirectional dilated window: attend to positions i +/- k*dilation
            mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
            for i in range(seq_len):
                # Left side
                left_positions = i - torch.arange(1, self.window_size) * self.dilation
                left_positions = left_positions[left_positions >= 0]
                # Right side
                right_positions = i + torch.arange(1, self.window_size) * self.dilation
                right_positions = right_positions[right_positions < seq_len]
                # Include current position
                mask[i, i] = 1
                mask[i, left_positions] = 1
                mask[i, right_positions] = 1

        # Add batch and head dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)

        return mask

    def forward(self, x, *, mask=None, **kwargs):
        """
        Computes dilated sliding window attention output.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim).
            mask (torch.Tensor, optional): Additional attention mask.

        Returns:
            torch.Tensor: The attention output.
        """
        batch, seq_len, emb_dim = x.shape
        h = self.num_heads
        head_dim = self.head_dim

        # Linear projections and reshape
        query = self.wq(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)
        key = self.wk(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)
        value = self.wv(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)

        if self.q_norm:
            query = self.q_norm(query)
        if self.k_norm:
            key = self.k_norm(key)

        # Create dilated window mask
        window_mask = self._create_dilated_window_mask(seq_len, x.device)

        # Combine with provided mask
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            combined_mask = window_mask & mask
        else:
            combined_mask = window_mask

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores.masked_fill(combined_mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)

        # Reshape and project
        attention_output = attention_output.transpose(1, 2).reshape(batch, seq_len, emb_dim)
        output = self.wo(attention_output)

        return output


@BlockRegistry.register()
class DynamicSlidingWindowAttention(SlidingWindowAttention):
    """
    Dynamic Sliding Window Attention with learnable window size per head.

    Extends sliding window attention with learnable window sizes for different
    attention heads.

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        window_size (int): Initial window size. Defaults to 128.
        learn_window (bool): If True, window size becomes learnable per head. Defaults to True.
        use_q_norm (bool): If True, applies RMSNorm to queries. Defaults to True.
        use_k_norm (bool): If True, applies RMSNorm to keys. Defaults to True.
        causal (bool): If True, applies causal masking within the window. Defaults to False.
    """

    def __init__(
        self,
        emb_dim=256,
        num_heads=4,
        window_size=128,
        learn_window=True,
        use_q_norm=True,
        use_k_norm=True,
        causal=False,
    ):
        super().__init__(emb_dim, num_heads, window_size, use_q_norm, use_k_norm, causal)

        self.learn_window = learn_window

        if learn_window:
            # Learnable window size parameter per head (logits for softmax)
            self.log_window_sizes = nn.Parameter(
                torch.log(torch.tensor(window_size, dtype=torch.float).repeat(num_heads))
            )

    def _get_window_sizes(self):
        """Get current window sizes for each head."""
        if self.learn_window:
            return torch.exp(self.log_window_sizes).long()
        else:
            return torch.tensor([self.window_size] * self.num_heads, device=self.log_window_sizes.device)

    def _create_dynamic_window_mask(self, seq_len, device):
        """
        Creates dynamic sliding window mask with per-head window sizes.

        Returns:
            torch.Tensor: Mask of shape (1, num_heads, seq_len, seq_len).
        """
        indices = torch.arange(seq_len, device=device)
        window_sizes = self._get_window_sizes()

        # Create mask for each head
        masks = []
        for window_size in window_sizes:
            if self.causal:
                mask = (indices.unsqueeze(0) - indices.unsqueeze(1) < window_size) & (
                    indices.unsqueeze(0) - indices.unsqueeze(1) >= 0
                )
            else:
                distance = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
                mask = distance < window_size

            masks.append(mask.unsqueeze(0))

        # Stack masks for all heads
        mask = torch.stack(masks, dim=1)  # (1, num_heads, seq_len, seq_len)

        return mask

    def forward(self, x, *, mask=None, **kwargs):
        """
        Computes dynamic sliding window attention output.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim).
            mask (torch.Tensor, optional): Additional attention mask.

        Returns:
            torch.Tensor: The attention output.
        """
        batch, seq_len, emb_dim = x.shape
        h = self.num_heads
        head_dim = self.head_dim

        # Linear projections and reshape
        query = self.wq(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)
        key = self.wk(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)
        value = self.wv(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)

        if self.q_norm:
            query = self.q_norm(query)
        if self.k_norm:
            key = self.k_norm(key)

        # Create dynamic window mask with per-head window sizes
        window_mask = self._create_dynamic_window_mask(seq_len, x.device)

        # Combine with provided mask
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            # Expand mask to match window_mask shape
            if mask.shape[1] == 1:
                mask = mask.expand(-1, h, -1, -1)
            combined_mask = window_mask & mask
        else:
            combined_mask = window_mask

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores.masked_fill(combined_mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)

        # Reshape and project
        attention_output = attention_output.transpose(1, 2).reshape(batch, seq_len, emb_dim)
        output = self.wo(attention_output)

        return output
