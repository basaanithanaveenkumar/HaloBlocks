import torch.nn as nn

from haloblocks.blocks.transformer.transformer_block import TransformerBlock
from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry


@BlockRegistry.register()
class DecoderTransformer(Block):
    def __init__(
        self,
        num_layers=16,
        emb_dim=1024,
        num_heads=32,
        mlp_dim=512,
        drop_fact=0.0,
        use_moe=True,
        moe_hid_scale=1.2,
        moe_num_routed_experts=16,
        moe_top_k=4,
        moe_num_shared_experts=2,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    drop_fact=drop_fact,
                    causal_mask=True,
                    use_moe=use_moe,
                    moe_hid_scale=moe_hid_scale,
                    moe_num_routed_experts=moe_num_routed_experts,
                    moe_top_k=moe_top_k,
                    moe_num_shared_experts=moe_num_shared_experts,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x
