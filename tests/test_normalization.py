import torch
import unittest
from haloblocks.core.factory import BlockFactory
from haloblocks.blocks.attention.utils import RMSNorm

class TestNormalization(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 8
        self.emb_dim = 16

    def test_rmsnorm_behavior(self):
        norm = RMSNorm(self.emb_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)
        output = norm(x)
        self.assertEqual(output.shape, x.shape)
        # Check that it's not the same as input
        self.assertFalse(torch.allclose(x, output))

    def test_self_attention_normalization(self):
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)
        
        # Without norm
        config1 = {'type': 'self_attention_basic', 'emb_dim': self.emb_dim, 'use_q_norm': False, 'use_k_norm': False}
        block1 = BlockFactory.create(config1)
        out1 = block1(x)
        
        # With norm
        config2 = {'type': 'self_attention_basic', 'emb_dim': self.emb_dim, 'use_q_norm': True, 'use_k_norm': True}
        block2 = BlockFactory.create(config2)
        # Copy weights to ensure difference is only from norm
        block2.query_proj.weight.data.copy_(block1.query_proj.weight.data)
        block2.key_proj.weight.data.copy_(block1.key_proj.weight.data)
        block2.val_proj.weight.data.copy_(block1.val_proj.weight.data)
        
        out2 = block2(x)
        self.assertFalse(torch.allclose(out1, out2))

    def test_gqa_normalization(self):
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)
        
        config = {
            'type': 'grouped_query_attention', 
            'emb_dim': self.emb_dim, 
            'num_heads': 4,
            'num_kv_heads': 2,
            'use_q_norm': True,
            'use_k_norm': True
        }
        block = BlockFactory.create(config)
        output = block(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.emb_dim))

if __name__ == "__main__":
    unittest.main()
