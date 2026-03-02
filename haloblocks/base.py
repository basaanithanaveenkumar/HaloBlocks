"""Base Block class for HaloBlocks."""

import torch.nn as nn


class Block(nn.Module):
    """Base class for all HaloBlocks components.

    Every model component (attention layer, head, encoder, decoder, etc.)
    extends this class. It provides a common interface and can be composed
    with other blocks to build full models.
    """

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement a forward() method."
        )

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Return the total number of parameters in this block.

        Args:
            trainable_only: If True, count only trainable parameters.

        Returns:
            Integer count of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        num_params = self.num_parameters()
        base = super().__repr__()
        return f"{base}\n  Total parameters: {num_params:,}"
