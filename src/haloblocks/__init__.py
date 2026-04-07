from haloblocks.core.block import Block
from haloblocks.core.builder import (
    StackedTransformerBlock,
    StackedTransformerBlocks,
    TransformerBlockBuilder,
)
from haloblocks.core.composite import CompositeBlock
from haloblocks.core.factory import BlockFactory
from haloblocks.core.registry import BlockRegistry

from . import blocks  # noqa: F401 — registry; exposes ``from haloblocks import blocks``

# Convenience exports
create = BlockFactory.create

__all__ = [
    "Block",
    "BlockRegistry",
    "BlockFactory",
    "CompositeBlock",
    "TransformerBlockBuilder",
    "StackedTransformerBlocks",
    "StackedTransformerBlock",
    "create",
    "blocks",
]


def main() -> None:
    print("HaloBlocks: A modular neural network component library.")
