import torch
import unittest
import math
import sys
import os

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from haloblocks.blocks.positional_embedding.sinusoidal import SinusoidalPositionalEmbedding
from haloblocks.blocks.positional_embedding.learned import LearnedPositionalEmbedding
from haloblocks.blocks.positional_embedding.rotary import RotaryPositionalEmbedding
from haloblocks.blocks.positional_embedding.alibi import AlibiPositionalBias

from haloblocks.core.factory import BlockFactory
# Ensure modules are imported for registry
import haloblocks.blocks.positional_embedding.sinusoidal
import haloblocks.blocks.positional_embedding.learned
import haloblocks.blocks.positional_embedding.rotary
import haloblocks.blocks.positional_embedding.alibi


class TestSinusoidalPositionalEmbedding(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.emb_dim = 64

    def test_output_shape(self):
        block = SinusoidalPositionalEmbedding(emb_dim=self.emb_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)
        output = block(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.emb_dim))

    def test_odd_emb_dim(self):
        emb_dim = 65
        block = SinusoidalPositionalEmbedding(emb_dim=emb_dim)
        x = torch.randn(self.batch_size, self.seq_len, emb_dim)
        output = block(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, emb_dim))

    def test_deterministic(self):
        """Fixed sinusoidal encoding should produce identical outputs for identical inputs."""
        block = SinusoidalPositionalEmbedding(emb_dim=self.emb_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)
        out1 = block(x)
        out2 = block(x)
        self.assertTrue(torch.allclose(out1, out2))

    def test_factory_create(self):
        config = {'type': 'sinusoidal_positional_embedding', 'emb_dim': self.emb_dim}
        block = BlockFactory.create(config)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)
        output = block(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.emb_dim))

    def test_with_dropout(self):
        block = SinusoidalPositionalEmbedding(emb_dim=self.emb_dim, dropout=0.1)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)
        block.train()
        output = block(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.emb_dim))


class TestLearnedPositionalEmbedding(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.emb_dim = 64
        self.max_len = 100

    def test_output_shape(self):
        block = LearnedPositionalEmbedding(emb_dim=self.emb_dim, max_len=self.max_len)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)
        output = block(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.emb_dim))

    def test_max_len_overflow(self):
        block = LearnedPositionalEmbedding(emb_dim=self.emb_dim, max_len=5)
        x = torch.randn(self.batch_size, 10, self.emb_dim)
        with self.assertRaises(ValueError):
            block(x)

    def test_factory_create(self):
        config = {
            'type': 'learned_positional_embedding',
            'emb_dim': self.emb_dim,
            'max_len': self.max_len,
        }
        block = BlockFactory.create(config)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)
        output = block(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.emb_dim))

    def test_learnable_params(self):
        block = LearnedPositionalEmbedding(emb_dim=self.emb_dim, max_len=self.max_len)
        param_count = sum(p.numel() for p in block.parameters() if p.requires_grad)
        self.assertEqual(param_count, self.max_len * self.emb_dim)


class TestRotaryPositionalEmbedding(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_heads = 8
        self.seq_len = 10
        self.head_dim = 32

    def test_output_shape(self):
        block = RotaryPositionalEmbedding(head_dim=self.head_dim)
        q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        q_rot, k_rot = block(q, k)
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)

    def test_different_qk_lengths(self):
        block = RotaryPositionalEmbedding(head_dim=self.head_dim)
        q = torch.randn(self.batch_size, self.num_heads, 5, self.head_dim)
        k = torch.randn(self.batch_size, self.num_heads, 10, self.head_dim)
        q_rot, k_rot = block(q, k)
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)

    def test_dynamic_cache_extension(self):
        block = RotaryPositionalEmbedding(head_dim=self.head_dim, max_len=16)
        q = torch.randn(self.batch_size, self.num_heads, 32, self.head_dim)
        k = torch.randn(self.batch_size, self.num_heads, 32, self.head_dim)
        # Should not error — cache dynamically extends
        q_rot, k_rot = block(q, k)
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)

    def test_factory_create(self):
        config = {'type': 'rotary_positional_embedding', 'head_dim': self.head_dim}
        block = BlockFactory.create(config)
        q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        q_rot, k_rot = block(q, k)
        self.assertEqual(q_rot.shape, q.shape)


class TestAlibiPositionalBias(unittest.TestCase):
    def setUp(self):
        self.num_heads = 8
        self.tgt_len = 10
        self.src_len = 10

    def test_output_shape(self):
        block = AlibiPositionalBias(num_heads=self.num_heads)
        bias = block(tgt_len=self.tgt_len, src_len=self.src_len)
        self.assertEqual(bias.shape, (self.num_heads, self.tgt_len, self.src_len))

    def test_non_power_of_2_heads(self):
        block = AlibiPositionalBias(num_heads=6)
        bias = block(tgt_len=self.tgt_len, src_len=self.src_len)
        self.assertEqual(bias.shape, (6, self.tgt_len, self.src_len))

    def test_bias_is_non_positive(self):
        """ALiBi bias should always be <= 0 (penalty for distance)."""
        block = AlibiPositionalBias(num_heads=self.num_heads)
        bias = block(tgt_len=self.tgt_len, src_len=self.src_len)
        self.assertTrue((bias <= 0).all())

    def test_diagonal_is_zero(self):
        """Self-position (distance 0) should have zero bias."""
        block = AlibiPositionalBias(num_heads=self.num_heads)
        bias = block(tgt_len=self.tgt_len, src_len=self.tgt_len)
        diagonal = torch.diagonal(bias, dim1=-2, dim2=-1)
        self.assertTrue(torch.allclose(diagonal, torch.zeros_like(diagonal)))

    def test_factory_create(self):
        config = {'type': 'alibi_positional_bias', 'num_heads': self.num_heads}
        block = BlockFactory.create(config)
        bias = block(tgt_len=self.tgt_len, src_len=self.src_len)
        self.assertEqual(bias.shape, (self.num_heads, self.tgt_len, self.src_len))

    def test_custom_slope_factor(self):
        block = AlibiPositionalBias(num_heads=self.num_heads, slope_factor=0.5)
        bias = block(tgt_len=self.tgt_len, src_len=self.src_len)
        self.assertEqual(bias.shape, (self.num_heads, self.tgt_len, self.src_len))


if __name__ == "__main__":
    unittest.main()
