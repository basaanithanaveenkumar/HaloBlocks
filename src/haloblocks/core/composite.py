import torch.nn as nn
from typing import List
from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

@BlockRegistry.register("composite_block")
class CompositeBlock(Block):
    """
    A container block that executes a sequence of sub-blocks.

    This block simplifies the creation of sequential pipelines by wrapping
    multiple Block instances and executing them in the order they are provided.

    Args:
        blocks (List[Block]): A list of Block instances to be executed sequentially.
    """

    def __init__(self, blocks: List[Block]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, **kwargs):
        """
        Sequentially passes the input through all registered sub-blocks.

        Args:
            x (torch.Tensor): Input tensor.
            **kwargs: Passed to each sub-block's forward method.

        Returns:
            torch.Tensor: The final output after passing through all sub-blocks.
        """
        for block in self.blocks:
            x = block(x, **kwargs)
        return x