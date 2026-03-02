"""Tests for decoder blocks."""

import torch

from haloblocks import TransformerDecoderLayerBlock, TransformerDecoderBlock


def test_decoder_layer_output_shape():
    layer = TransformerDecoderLayerBlock(d_model=32, num_heads=4, d_ff=64)
    x = torch.randn(2, 5, 32)
    memory = torch.randn(2, 7, 32)
    out, sa_w, ca_w = layer(x, memory)
    assert out.shape == (2, 5, 32)
    assert sa_w.shape == (2, 4, 5, 5)
    assert ca_w.shape == (2, 4, 5, 7)


def test_decoder_layer_with_masks():
    layer = TransformerDecoderLayerBlock(d_model=16, num_heads=2, d_ff=32)
    x = torch.randn(2, 4, 16)
    memory = torch.randn(2, 6, 16)
    tgt_mask = torch.zeros(2, 4, dtype=torch.bool)
    mem_mask = torch.zeros(2, 6, dtype=torch.bool)
    mem_mask[:, 5] = True
    out, _, _ = layer(x, memory, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=mem_mask)
    assert out.shape == (2, 4, 16)


def test_decoder_block_output_shape():
    decoder = TransformerDecoderBlock(d_model=32, num_heads=4, d_ff=64, num_layers=3)
    x = torch.randn(2, 5, 32)
    memory = torch.randn(2, 7, 32)
    out = decoder(x, memory)
    assert out.shape == (2, 5, 32)


def test_decoder_block_num_layers():
    decoder = TransformerDecoderBlock(d_model=16, num_heads=2, d_ff=32, num_layers=4)
    assert len(decoder.layers) == 4


def test_decoder_causal_self_attention():
    """Decoder self-attention must be causal."""
    decoder = TransformerDecoderBlock(d_model=16, num_heads=2, d_ff=32, num_layers=1)
    assert decoder.layers[0].self_attn.causal is True
