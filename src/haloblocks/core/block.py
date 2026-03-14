from abc import ABC, abstractmethod
import torch.nn as nn

class Block(ABC, nn.Module):
    """
    Base class for all composable neural network components in HaloBlocks.

    Every model component (e.g., Attention, MoE, Transformer Layer) should
    inherit from this class to ensure compatibility with the BlockFactory
    and BlockRegistry.
    """

    def __init__(self):
        super().__init__()
        self._name = self.__class__.__name__

    @abstractmethod
    def forward(self, x, **kwargs):
        """
        Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The main input tensor.
            **kwargs: Additional arguments such as masks, key-value caches, etc.

        Returns:
            torch.Tensor or tuple: The transformed tensor(s).
        """
        pass

    @property
    def name(self):
        """Returns the name of the block."""
        return self._name

    def extra_repr(self):
        return f"name={self.name}"