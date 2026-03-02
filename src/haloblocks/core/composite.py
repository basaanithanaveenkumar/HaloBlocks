from typing import List

class CompositeBlock(Block):
    """A block that sequentially executes its children."""

    def __init__(self, blocks: List[Block]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, **kwargs):
        for block in self.blocks:
            x = block(x, **kwargs)
        return x