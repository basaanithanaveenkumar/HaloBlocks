import torch
import unittest
import math
import sys
import os

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from haloblocks.core.factory import BlockFactory

# Explicitly import blocks to ensure they are registered
import haloblocks.blocks.attention.self_attention
import haloblocks.blocks.moe.deepseek_moe
import haloblocks.blocks.transformer.transformer_block
import haloblocks.blocks.transformer.decoder
import haloblocks.blocks.vla.flow_decoder
import haloblocks.blocks.attention.scaledotprod

class TestHaloBlocks(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.emb_dim = 64

    def test_self_attn(self):
        config = {'type': 'self_attention_basic', 'emb_dim': self.emb_dim}
        block = BlockFactory.create(config)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)
        output = block(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.emb_dim))

    def test_multi_head_attn(self):
        config = {'type': 'multi_head_attn', 'emb_dim': self.emb_dim, 'num_heads': 8}
        block = BlockFactory.create(config)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)
        output = block(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.emb_dim))

    def test_deepseek_moe(self):
        config = {
            'type': 'deepseek_moe',
            'emb_dim': self.emb_dim,
            'hid_dim': 128,
            'num_router_exprts': 4,
            'best_k': 2,
            'num_shared_exprts': 1
        }
        block = BlockFactory.create(config)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)
        output = block(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.emb_dim))

    def test_transformer_block(self):
        # Test with MoE
        config_moe = {
            'type': 'transformer_block',
            'emb_dim': self.emb_dim,
            'num_heads': 8,
            'use_moe': True
        }
        block_moe = BlockFactory.create(config_moe)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)
        output_moe = block_moe(x)
        self.assertEqual(output_moe.shape, (self.batch_size, self.seq_len, self.emb_dim))

        # Test with standard MLP
        config_mlp = {
            'type': 'transformer_block',
            'emb_dim': self.emb_dim,
            'num_heads': 8,
            'use_moe': False,
            'mlp_dim': 128
        }
        block_mlp = BlockFactory.create(config_mlp)
        output_mlp = block_mlp(x)
        self.assertEqual(output_mlp.shape, (self.batch_size, self.seq_len, self.emb_dim))

    def test_decoder_transformer(self):
        config = {
            'type': 'decoder_transformer',
            'num_layers': 2,
            'emb_dim': self.emb_dim,
            'num_heads': 8
        }
        block = BlockFactory.create(config)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)
        output = block(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.emb_dim))

    def test_vla_flow_decoder(self):
        obs_dim = 128
        action_dim_flat = 32
        config = {
            'type': 'vla_flow_decoder',
            'action_dim_flat': action_dim_flat,
            'obs_dim': obs_dim,
            'hidden_dim': 64,
            'time_embed_dim': 32
        }
        block = BlockFactory.create(config)
        x_t = torch.randn(self.batch_size, action_dim_flat)
        t = torch.rand(self.batch_size)
        cond = torch.randn(self.batch_size, obs_dim)
        output = block(x_t, t, cond)
        self.assertEqual(output.shape, (self.batch_size, action_dim_flat))

    def test_sinusoidal_time_embedding(self):
        from haloblocks.blocks.vla.flow_decoder import SinusoidalTimeEmbedding
        embed_dim = 32
        block = SinusoidalTimeEmbedding(embed_dim)
        t = torch.rand(self.batch_size)
        output = block(t)
        self.assertEqual(output.shape, (self.batch_size, embed_dim))

    def test_scaled_dot_product_attention(self):
        config = {'type': 'scaled_dot_product_attention', 'dropout': 0.0}
        block = BlockFactory.create(config)
        q = torch.randn(2, 8, 10, 32)
        k = torch.randn(2, 8, 10, 32)
        v = torch.randn(2, 8, 10, 32)
        output = block(q, k, v)
        self.assertEqual(output.shape, (2, 8, 10, 32))

if __name__ == "__main__":
    unittest.main()
