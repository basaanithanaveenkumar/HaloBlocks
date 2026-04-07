"""Attention score masking helpers (not normalization)."""

import torch


def check_attention_mask_broadcasts(scores: torch.Tensor, mask: torch.Tensor, *, name: str = "mask") -> None:
    """Raise ValueError if ``mask`` cannot broadcast to ``scores`` (attention logits)."""
    try:
        torch.broadcast_shapes(scores.shape, mask.shape)
    except RuntimeError as e:
        raise ValueError(
            f"{name} with shape {tuple(mask.shape)} cannot broadcast to attention scores "
            f"with shape {tuple(scores.shape)}"
        ) from e
