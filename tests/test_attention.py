"""Tests for attention blocks."""

import pytest
import torch

from haloblocks import (
    MultiHeadAttentionBlock,
    SelfAttentionBlock,
    CrossAttentionBlock,
)


def test_multihead_attention_output_shape():
    block = MultiHeadAttentionBlock(d_model=32, num_heads=4)
    q = torch.randn(2, 5, 32)
    k = torch.randn(2, 7, 32)
    v = torch.randn(2, 7, 32)
    out, weights = block(q, k, v)
    assert out.shape == (2, 5, 32)
    assert weights.shape == (2, 4, 5, 7)


def test_multihead_attention_invalid_heads():
    with pytest.raises(ValueError):
        MultiHeadAttentionBlock(d_model=32, num_heads=5)


def test_multihead_attention_with_key_padding_mask():
    block = MultiHeadAttentionBlock(d_model=16, num_heads=2)
    q = torch.randn(2, 4, 16)
    kv = torch.randn(2, 6, 16)
    # Mask out last two positions in the key sequence
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 4:] = True
    out, _ = block(q, kv, kv, key_padding_mask=mask)
    assert out.shape == (2, 4, 16)


def test_self_attention_output_shape():
    block = SelfAttentionBlock(d_model=32, num_heads=4)
    x = torch.randn(2, 5, 32)
    out, weights = block(x)
    assert out.shape == (2, 5, 32)
    assert weights.shape == (2, 4, 5, 5)


def test_self_attention_causal():
    block = SelfAttentionBlock(d_model=16, num_heads=2, causal=True)
    x = torch.randn(1, 4, 16)
    out, weights = block(x)
    assert out.shape == (1, 4, 16)
    # Upper-triangular entries (future positions) should have near-zero weight
    assert weights[0, 0, 0, 1].item() < 1e-6


def test_cross_attention_output_shape():
    block = CrossAttentionBlock(d_model=32, num_heads=4)
    query = torch.randn(2, 3, 32)
    context = torch.randn(2, 7, 32)
    out, weights = block(query, context)
    assert out.shape == (2, 3, 32)
    assert weights.shape == (2, 4, 3, 7)


def test_cross_attention_with_padding_mask():
    block = CrossAttentionBlock(d_model=16, num_heads=2)
    query = torch.randn(2, 3, 16)
    context = torch.randn(2, 5, 16)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[0, 4] = True
    out, _ = block(query, context, key_padding_mask=mask)
    assert out.shape == (2, 3, 16)
