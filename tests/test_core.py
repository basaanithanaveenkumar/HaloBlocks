import os
import sys
import unittest

import torch

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from haloblocks.core.block import Block
from haloblocks.core.composite import CompositeBlock
from haloblocks.core.factory import BlockFactory
from haloblocks.core.registry import BlockRegistry


class MockBlock(Block):
    def __init__(self, val=1):
        super().__init__()
        self.val = val

    def forward(self, x, **kwargs):
        return x + self.val


BlockRegistry.register("mock_block")(MockBlock)


class TestCore(unittest.TestCase):
    def test_block_registry(self):
        self.assertEqual(BlockRegistry.get("mock_block"), MockBlock)
        self.assertEqual(BlockRegistry.get("CompositeBlock"), CompositeBlock)

    def test_block_factory_simple(self):
        config = {"type": "mock_block", "val": 5}
        block = BlockFactory.create(config)
        self.assertIsInstance(block, MockBlock)
        self.assertEqual(block.val, 5)

        x = torch.tensor(1.0)
        self.assertEqual(block(x), 6.0)

    def test_composite_block(self):
        b1 = MockBlock(1)
        b2 = MockBlock(10)
        cb = CompositeBlock([b1, b2])

        x = torch.tensor(0.0)
        # 0 + 1 + 10 = 11
        self.assertEqual(cb(x), 11.0)

    def test_block_factory_recursive(self):
        # Test creating a CompositeBlock via factory
        config = {
            "type": "CompositeBlock",
            "blocks": [{"type": "mock_block", "val": 1}, {"type": "mock_block", "val": 2}],
        }
        cb = BlockFactory.create(config)
        self.assertIsInstance(cb, CompositeBlock)
        self.assertEqual(len(cb.blocks), 2)

        x = torch.tensor(0.0)
        self.assertEqual(cb(x), 3.0)

    def test_block_properties(self):
        block = MockBlock()
        self.assertEqual(block.name, "MockBlock")
        self.assertIn("name=MockBlock", block.extra_repr())


if __name__ == "__main__":
    unittest.main()
