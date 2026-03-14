from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry
from haloblocks.core.factory import BlockFactory
from haloblocks.core.composite import CompositeBlock
from haloblocks import blocks  # Trigger registration

# Convenience exports
create = BlockFactory.create

__all__ = ["Block", "BlockRegistry", "BlockFactory", "CompositeBlock", "create"]

def main() -> None:
    print("HaloBlocks: A modular neural network component library.")
