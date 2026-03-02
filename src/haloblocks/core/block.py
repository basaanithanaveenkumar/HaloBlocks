from abc import ABC, abstractmethod
import torch.nn as nn

class Block(ABC, nn.Module):
    """Base class for all composable model components."""

    def __init__(self):
        super().__init__()
        self._name = self.__class__.__name__

    @abstractmethod
    def forward(self, x, **kwargs):
        """Forward pass; kwargs allow passing masks, cache, etc."""
        pass

    @property
    def name(self):
        return self._name

    def extra_repr(self):
        return f"name={self.name}"