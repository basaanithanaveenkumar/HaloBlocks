"""Tests for the TransformerBlockBuilder (core/builder.py)."""

import pytest
import torch

import haloblocks as hb
from haloblocks import blocks

# ── 1. Defaults ──────────────────────────────────────────────────────


def test_default_mha_mlp():
    """Build with defaults → MHA + MLP, verify shapes."""
    block = hb.create("TransformerBlockBuilder", emb_dim=128)
    x = torch.randn(2, 16, 128)
    out = block(x)
    assert out.shape == x.shape


def test_default_forward_deterministic_eval():
    """In eval mode, two forward passes produce the same output."""
    block = hb.create("TransformerBlockBuilder", emb_dim=64)
    block.eval()
    x = torch.randn(1, 8, 64)
    assert torch.allclose(block(x), block(x))


# ── 2. Custom attention via config dict ──────────────────────────────


def test_gqa_via_config_dict():
    """Pass GQA as a config dict."""
    block = hb.create(
        "TransformerBlockBuilder",
        emb_dim=256,
        attn={"type": "GroupedQueryAttention", "num_heads": 8, "num_kv_heads": 2},
    )
    x = torch.randn(2, 16, 256)
    out = block(x)
    assert out.shape == x.shape


def test_sliding_window_via_config_dict():
    """Pass sliding-window attention as a config dict."""
    block = hb.create(
        "TransformerBlockBuilder",
        emb_dim=128,
        attn={"type": "SlidingWindowAttention", "num_heads": 4, "window_size": 8},
    )
    x = torch.randn(2, 32, 128)
    out = block(x)
    assert out.shape == x.shape


def test_gated_attention_via_config_dict():
    """Pass gated attention as a config dict."""
    block = hb.create(
        "TransformerBlockBuilder",
        emb_dim=128,
        attn={"type": "GatedAttention", "num_heads": 4},
    )
    x = torch.randn(2, 16, 128)
    out = block(x)
    assert out.shape == x.shape


# ── 3. Custom FFN via config dict ────────────────────────────────────


def test_moe_ffn_via_config_dict():
    """Use DeepSeek MoE as the FFN."""
    block = hb.create(
        "TransformerBlockBuilder",
        emb_dim=128,
        ffn={
            "type": "DeepseekMoE",
            "hid_dim": 256,
            "num_router_exprts": 4,
            "best_k": 2,
            "num_shared_exprts": 1,
        },
    )
    x = torch.randn(2, 8, 128)
    out = block(x)
    assert out.shape == x.shape


# ── 4. Pre-built Block instances ─────────────────────────────────────


def test_prebuilt_block_instances():
    """Pass pre-built Block objects instead of configs."""
    my_attn = blocks.GatedAttention(emb_dim=128, num_heads=4)
    my_ffn = blocks.MLP(input_dim=128, hidden_dims=[256], output_dim=128, activation="gelu")

    block = hb.create("TransformerBlockBuilder", emb_dim=128, attn=my_attn, ffn=my_ffn)
    x = torch.randn(2, 16, 128)
    out = block(x)
    assert out.shape == x.shape


# ── 5. Norm options ──────────────────────────────────────────────────


def test_rmsnorm():
    """Build with RMSNorm instead of LayerNorm."""
    block = hb.create("TransformerBlockBuilder", emb_dim=64, norm="rmsnorm")
    x = torch.randn(2, 8, 64)
    out = block(x)
    assert out.shape == x.shape


def test_invalid_norm_raises():
    with pytest.raises(ValueError, match="Unknown norm type"):
        hb.create("TransformerBlockBuilder", emb_dim=64, norm="batchnorm")


# ── 6. Residual dropout ─────────────────────────────────────────────


def test_dropout():
    """Verify dropout produces different outputs in train mode."""
    block = hb.create("TransformerBlockBuilder", emb_dim=64, drop_fact=0.5)
    block.train()
    x = torch.randn(4, 16, 64)
    # With high dropout, repeated forward passes should differ (with overwhelming probability)
    out1 = block(x)
    out2 = block(x)
    # They *could* be equal by cosmic chance, but at p=0.5 on 4*16*64 elements, never.
    assert not torch.allclose(out1, out2)


# ── 7. stack() ───────────────────────────────────────────────────────


def test_stack_shapes():
    """stack(4) produces a StackedTransformerBlocks with correct output shape."""
    single = hb.create("TransformerBlockBuilder", emb_dim=128)
    stacked = single.stack(4)
    x = torch.randn(2, 16, 128)
    out = stacked(x)
    assert out.shape == x.shape


def test_stack_independent_weights():
    """Each layer in the stack has its own parameters (not shared)."""
    single = hb.create("TransformerBlockBuilder", emb_dim=64)
    stacked = single.stack(3)
    # Verify different layers have different parameter objects
    layers_list = list(stacked.layers)
    p0 = list(layers_list[0].parameters())[0]
    p1 = list(layers_list[1].parameters())[0]
    assert p0.data_ptr() != p1.data_ptr()


# ── 8. Full config-dict creation (JSON-like) ────────────────────────


def test_full_config_dict():
    """Create entirely from a config dict (as if loaded from JSON)."""
    config = {
        "type": "TransformerBlockBuilder",
        "emb_dim": 128,
        "attn": {"type": "MultiHeadAttention", "num_heads": 4},
        "ffn": {"type": "MLP", "input_dim": 128, "hidden_dims": [256], "output_dim": 128},
        "norm": "layernorm",
        "drop_fact": 0.0,
    }
    block = hb.create(config)
    x = torch.randn(2, 16, 128)
    out = block(x)
    assert out.shape == x.shape


# ── 9. attn_kwargs shorthand ─────────────────────────────────────────


def test_attn_kwargs_string_key():
    """Pass attention as a string key + attn_kwargs for extra params."""
    block = hb.create(
        "TransformerBlockBuilder",
        emb_dim=128,
        attn="GatedAttention",
        attn_kwargs={"num_heads": 4},
    )
    x = torch.randn(2, 8, 128)
    out = block(x)
    assert out.shape == x.shape


# ── 10. Backward pass ───────────────────────────────────────────────


def test_gradients_flow():
    """Ensure the full graph is differentiable."""
    block = hb.create(
        "TransformerBlockBuilder",
        emb_dim=64,
        attn={"type": "GroupedQueryAttention", "num_heads": 4, "num_kv_heads": 2},
        ffn={"type": "DeepseekMoE", "hid_dim": 128, "num_router_exprts": 4, "best_k": 2, "num_shared_exprts": 1},
    )
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# ── stacked_transformer_block (config / composite) ───────────────────


def test_stacked_transformer_block_factory():
    """Registry type builds N layers; same behavior as custom_transformer_block.stack(N)."""
    block = hb.create("StackedTransformerBlock", emb_dim=128, num_layers=4)
    assert len(block.stacked.layers) == 4
    x = torch.randn(2, 16, 128)
    assert block(x).shape == x.shape


def test_stacked_transformer_block_in_composite():
    """composite_block can nest stacked_transformer_block via BlockFactory."""
    model = hb.create(
        {
            "type": "CompositeBlock",
            "blocks": [
                {
                    "type": "StackedTransformerBlock",
                    "emb_dim": 64,
                    "num_layers": 2,
                    "drop_fact": 0.0,
                },
                {
                    "type": "MLP",
                    "input_dim": 64,
                    "hidden_dims": [32],
                    "output_dim": 10,
                    "activation": "gelu",
                },
            ],
        }
    )
    x = torch.randn(2, 5, 64)
    y = model(x)
    assert y.shape == (2, 5, 10)
