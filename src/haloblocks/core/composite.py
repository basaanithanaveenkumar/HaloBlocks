import torch.nn as nn
from typing import List
from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

@BlockRegistry.register("composite_block")
class CompositeBlock(Block):
    """A block that sequentially executes its children."""

    def __init__(self, blocks: List[Block]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, **kwargs):
        for block in self.blocks:
            x = block(x, **kwargs)
        return x