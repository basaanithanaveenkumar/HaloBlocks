import torch.nn as nn

from haloblocks.blocks.attention.self_attention import MultiHeadAttention
from haloblocks.blocks.moe.deepseek_moe import DeepseekMoE
from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry


@BlockRegistry.register()
class TransformerBlock(Block):
    def __init__(
        self,
        emb_dim=256,
        num_heads=8,
        mlp_dim=512,
        drop_fact=0.0,
        causal_mask=False,
        use_moe=True,
        moe_hid_scale=1.2,
        moe_num_routed_experts=16,
        moe_top_k=4,
        moe_num_shared_experts=2,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            emb_dim=emb_dim, num_heads=num_heads, drop_fact=drop_fact, causal_mask=causal_mask
        )
        self.norm1 = nn.LayerNorm(emb_dim)

        self.use_moe = use_moe
        if use_moe:
            moe_hid_dim = round(emb_dim * moe_hid_scale)
            self.ffn = DeepseekMoE(
                emb_dim,
                moe_hid_dim,
                num_router_exprts=moe_num_routed_experts,
                best_k=moe_top_k,
                num_shared_exprts=moe_num_shared_experts,
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(emb_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, emb_dim),
                nn.Dropout(drop_fact),
            )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x, **kwargs):
        attn_output = self.attn(self.norm1(x))
        x = x + attn_output

        ffn_output = self.ffn(self.norm2(x))
        return x + ffn_output
