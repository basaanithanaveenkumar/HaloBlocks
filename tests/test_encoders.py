"""Tests for encoder blocks."""

import torch

from haloblocks import TransformerEncoderLayerBlock, TransformerEncoderBlock


def test_encoder_layer_output_shape():
    layer = TransformerEncoderLayerBlock(d_model=32, num_heads=4, d_ff=64)
    x = torch.randn(2, 5, 32)
    out = layer(x)
    assert out.shape == (2, 5, 32)


def test_encoder_layer_with_padding_mask():
    layer = TransformerEncoderLayerBlock(d_model=16, num_heads=2, d_ff=32)
    x = torch.randn(2, 6, 16)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, 5] = True
    out = layer(x, src_key_padding_mask=mask)
    assert out.shape == (2, 6, 16)


def test_encoder_block_output_shape():
    encoder = TransformerEncoderBlock(d_model=32, num_heads=4, d_ff=64, num_layers=3)
    x = torch.randn(2, 5, 32)
    out = encoder(x)
    assert out.shape == (2, 5, 32)


def test_encoder_block_num_layers():
    encoder = TransformerEncoderBlock(d_model=16, num_heads=2, d_ff=32, num_layers=4)
    assert len(encoder.layers) == 4


def test_encoder_block_with_padding_mask():
    encoder = TransformerEncoderBlock(d_model=16, num_heads=2, d_ff=32, num_layers=2)
    x = torch.randn(3, 4, 16)
    mask = torch.zeros(3, 4, dtype=torch.bool)
    mask[:, 3] = True
    out = encoder(x, src_key_padding_mask=mask)
    assert out.shape == (3, 4, 16)
