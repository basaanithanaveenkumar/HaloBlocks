"""Tests for layer blocks."""

import pytest
import torch

from haloblocks import (
    Block,
    LinearBlock,
    LayerNormBlock,
    DropoutBlock,
    FeedForwardBlock,
    EmbeddingBlock,
)


def test_block_is_abstract():
    block = Block()
    with pytest.raises(NotImplementedError):
        block.forward(torch.zeros(1))


def test_linear_block_shape():
    block = LinearBlock(16, 32)
    x = torch.randn(2, 16)
    out = block(x)
    assert out.shape == (2, 32)


def test_linear_block_no_bias():
    block = LinearBlock(8, 4, bias=False)
    assert block.linear.bias is None


def test_linear_block_num_parameters():
    block = LinearBlock(8, 4)  # 8*4 weights + 4 bias = 36
    assert block.num_parameters() == 36


def test_layernorm_block_shape():
    block = LayerNormBlock(16)
    x = torch.randn(2, 5, 16)
    out = block(x)
    assert out.shape == (2, 5, 16)


def test_dropout_block_shape():
    block = DropoutBlock(p=0.0)
    x = torch.randn(3, 10)
    out = block(x)
    assert out.shape == (3, 10)
    assert torch.allclose(out, x)


def test_feedforward_block_shape():
    block = FeedForwardBlock(d_model=32, d_ff=64)
    x = torch.randn(2, 5, 32)
    out = block(x)
    assert out.shape == (2, 5, 32)


def test_feedforward_block_relu():
    block = FeedForwardBlock(d_model=16, d_ff=32, activation="relu")
    x = torch.randn(1, 3, 16)
    out = block(x)
    assert out.shape == (1, 3, 16)


def test_feedforward_block_invalid_activation():
    with pytest.raises(ValueError):
        FeedForwardBlock(d_model=16, d_ff=32, activation="tanh")


def test_embedding_block_shape():
    block = EmbeddingBlock(vocab_size=100, d_model=16)
    idx = torch.randint(0, 100, (2, 5))
    out = block(idx)
    assert out.shape == (2, 5, 16)


def test_num_parameters_trainable_only():
    block = LinearBlock(8, 4)
    for p in block.parameters():
        p.requires_grad = False
    assert block.num_parameters(trainable_only=True) == 0
