import pytest
import torch
import torch.nn as nn

from haloblocks.blocks import attention
from haloblocks.blocks.attention import (
    CrossAttention,
    GroupedQueryAttention,
    MultiHeadLatentAttention,
    ScaledDotProductAttention,
    SelfAttention,
    TrinityAttention,
    TrinityCrossAttention,
    multi_query_attention,
)
from haloblocks.blocks.attention.masking import check_attention_mask_broadcasts


def test_attention_package_namespace_aliases_classes():
    """``from haloblocks.blocks import attention`` exposes the same classes as submodules."""
    assert attention.MultiQueryAttention is multi_query_attention.MultiQueryAttention
    assert attention.GroupedQueryAttention is GroupedQueryAttention
    assert attention.SelfAttention is SelfAttention


def test_attention_mechanisms():
    batch_size = 2
    seq_len = 8
    emb_dim = 256
    num_heads = 8

    query = torch.randn(batch_size, seq_len, emb_dim)
    key = torch.randn(batch_size, seq_len, emb_dim)
    value = torch.randn(batch_size, seq_len, emb_dim)

    self_attn = SelfAttention(emb_dim, return_attn_weights=False, use_q_norm=False, use_k_norm=False)
    self_attn_output = self_attn(query)
    assert self_attn_output.shape == (batch_size, seq_len, emb_dim)

    scaled_attn = ScaledDotProductAttention(
        dropout=0.0, head_dim=emb_dim // num_heads, use_q_norm=False, use_k_norm=False
    )
    scaled_output = scaled_attn(query, key, value)
    assert scaled_output.shape == (batch_size, seq_len, emb_dim)

    trinity_attn = TrinityAttention(emb_dim=emb_dim, num_heads=num_heads, use_q_norm=False, use_k_norm=False)
    trinity_output = trinity_attn(query)
    assert trinity_output.shape == (batch_size, seq_len, emb_dim)

    cross_attn = CrossAttention(emb_dim, return_attn_weights=False, use_q_norm=False, use_k_norm=False)
    cross_output = cross_attn(query, key)
    assert cross_output.shape == (batch_size, seq_len, emb_dim)
    cross_output_sep = cross_attn(query, key, value)
    assert cross_output_sep.shape == (batch_size, seq_len, emb_dim)

    mqa_attn = attention.MultiQueryAttention(emb_dim=emb_dim, num_heads=num_heads)
    mqa_output = mqa_attn(query, context=key, value_context=value)
    assert mqa_output.shape == (batch_size, seq_len, emb_dim)

    gqa_attn = GroupedQueryAttention(emb_dim, num_heads)
    gqa_output = gqa_attn(query, context=key)
    assert gqa_output.shape == (batch_size, seq_len, emb_dim)

    mla_attn = MultiHeadLatentAttention(emb_dim, num_heads)
    mla_output = mla_attn(query, context=key, value_context=value)
    assert mla_output.shape == (batch_size, seq_len, emb_dim)


def test_create_attention():
    batch_size = 2
    seq_len = 8
    emb_dim = 256
    num_heads = 8

    query = torch.randn(batch_size, seq_len, emb_dim)
    key = torch.randn(batch_size, seq_len, emb_dim)
    value = torch.randn(batch_size, seq_len, emb_dim)

    attentions = {
        "self": SelfAttention(emb_dim, return_attn_weights=False, use_q_norm=False, use_k_norm=False),
        "scaled": ScaledDotProductAttention(
            dropout=0.0, head_dim=emb_dim // num_heads, use_q_norm=False, use_k_norm=False
        ),
        "trinity": TrinityAttention(emb_dim=emb_dim, num_heads=num_heads, use_q_norm=False, use_k_norm=False),
        "cross": CrossAttention(emb_dim, return_attn_weights=False, use_q_norm=False, use_k_norm=False),
        "mqa": attention.MultiQueryAttention(emb_dim=emb_dim, num_heads=num_heads),
        "gqa": GroupedQueryAttention(emb_dim, num_heads),
        "mha": MultiHeadLatentAttention(emb_dim, num_heads),
    }

    for name, attn in attentions.items():
        if name in ("self", "trinity"):
            output = attn(query)
        elif name == "cross":
            output = attn(query, key)
        elif name == "gqa":
            output = attn(query, context=key)
        elif name == "scaled":
            output = attn(query, key, value)
        else:
            output = attn(query, context=key, value_context=value)
        assert output.shape == (batch_size, seq_len, emb_dim)


def test_attention_in_simple_module():
    batch_size = 2
    seq_len = 8
    emb_dim = 256
    num_heads = 8

    query = torch.randn(batch_size, seq_len, emb_dim)

    class SimpleLayer(nn.Module):
        def __init__(self, emb_dim, num_heads):
            super().__init__()
            self.self_attn = SelfAttention(emb_dim, return_attn_weights=False, use_q_norm=False, use_k_norm=False)
            self.norm = nn.LayerNorm(emb_dim)
            self.ffn = nn.Sequential(
                nn.Linear(emb_dim, 4 * emb_dim),
                nn.GELU(),
                nn.Linear(4 * emb_dim, emb_dim),
            )

        def forward(self, x):
            attn_output = self.self_attn(x)
            x = self.norm(attn_output + x)
            ffn_output = self.ffn(x)
            return self.norm(ffn_output + x)

    layer = SimpleLayer(emb_dim, num_heads)
    output = layer(query)
    assert output.shape == (batch_size, seq_len, emb_dim)


def test_composite_attention():
    batch_size = 2
    seq_len = 8
    emb_dim = 256
    num_heads = 8

    query = torch.randn(batch_size, seq_len, emb_dim)
    key = torch.randn(batch_size, seq_len, emb_dim)
    value = torch.randn(batch_size, seq_len, emb_dim)

    class CompositeAttention(nn.Module):
        def __init__(self, emb_dim, num_heads):
            super().__init__()
            self.self_attn = SelfAttention(emb_dim, return_attn_weights=False, use_q_norm=False, use_k_norm=False)
            self.cross_attn = CrossAttention(emb_dim, return_attn_weights=False, use_q_norm=False, use_k_norm=False)
            self.mla_attn = MultiHeadLatentAttention(emb_dim, num_heads)

        def forward(self, query, key, value):
            self_out = self.self_attn(query)
            cross_out = self.cross_attn(query, key)
            mla_out = self.mla_attn(query, context=key, value_context=value)
            return (self_out + cross_out + mla_out) / 3

    composite = CompositeAttention(emb_dim, num_heads)
    output = composite(query, key, value)
    assert output.shape == (batch_size, seq_len, emb_dim)


def test_attention_with_blocks():
    batch_size = 2
    seq_len = 8
    emb_dim = 256
    num_heads = 8

    query = torch.randn(batch_size, seq_len, emb_dim)
    key = torch.randn(batch_size, seq_len, emb_dim)
    value = torch.randn(batch_size, seq_len, emb_dim)

    class TransformerBlock(nn.Module):
        def __init__(self, emb_dim, num_heads):
            super().__init__()
            self.self_attn = SelfAttention(emb_dim, return_attn_weights=False, use_q_norm=False, use_k_norm=False)
            self.cross_attn = CrossAttention(emb_dim, return_attn_weights=False, use_q_norm=False, use_k_norm=False)
            self.mla_attn = MultiHeadLatentAttention(emb_dim, num_heads)
            self.norm1 = nn.LayerNorm(emb_dim)
            self.norm2 = nn.LayerNorm(emb_dim)
            self.ffn = nn.Sequential(
                nn.Linear(emb_dim, 4 * emb_dim),
                nn.GELU(),
                nn.Linear(4 * emb_dim, emb_dim),
            )

        def forward(self, x, key, value):
            self_out = self.self_attn(x)
            x = self.norm1(self_out + x)
            cross_out = self.cross_attn(x, key)
            x = self.norm2(cross_out + x)
            mla_out = self.mla_attn(x, context=key, value_context=value)
            x = self.norm2(mla_out + x)
            ffn_out = self.ffn(x)
            return self.norm2(ffn_out + x)

    block = TransformerBlock(emb_dim, num_heads)
    output = block(query, key, value)
    assert output.shape == (batch_size, seq_len, emb_dim)


def test_gqa_three_positional_args_rejected():
    q = torch.randn(2, 8, 64)
    k = torch.randn(2, 8, 64)
    v = torch.randn(2, 8, 64)
    mod = GroupedQueryAttention(64, num_heads=8)
    with pytest.raises(TypeError):
        mod(q, k, v)


def test_invalid_attention_mask_raises():
    batch_size, seq_len, emb_dim = 2, 8, 256
    num_heads = 8
    q = torch.randn(batch_size, seq_len, emb_dim)
    k = torch.randn(batch_size, seq_len, emb_dim)
    gqa_mod = GroupedQueryAttention(emb_dim, num_heads=num_heads)
    scores = torch.randn(batch_size, num_heads, seq_len, seq_len)
    bad = torch.randn(3, 3)
    with pytest.raises(ValueError, match="mask"):
        check_attention_mask_broadcasts(scores, bad, name="mask")
    with pytest.raises(ValueError, match="mask"):
        gqa_mod(q, k, mask=bad)


def test_trinity_cross_attention_separate_value():
    emb_dim, n_heads = 64, 4
    q = torch.randn(2, 5, emb_dim)
    ctx = torch.randn(2, 7, emb_dim)
    val = torch.randn(2, 7, emb_dim)
    m = TrinityCrossAttention(emb_dim=emb_dim, num_heads=n_heads, use_q_norm=False, use_k_norm=False)
    out = m(q, ctx, val)
    assert out.shape == (2, 5, emb_dim)


if __name__ == "__main__":
    pytest.main([__file__])
