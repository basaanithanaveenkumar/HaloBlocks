# HaloBlocks

A Python library where all model components are treated as composable **blocks** — including heads, attention layers, feed-forward layers, encoders, and decoders.

Every block is a `torch.nn.Module` that can be mixed and matched to build custom neural network architectures.

## Installation

```bash
pip install haloblocks
```

Or from source:

```bash
git clone https://github.com/basaanithanaveenkumar/HaloBlocks.git
cd HaloBlocks
pip install -e .
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 1.13

## Quick Start

```python
import torch
import haloblocks as hb

# --- Layers ---
linear  = hb.LinearBlock(128, 256)
norm    = hb.LayerNormBlock(256)
ff      = hb.FeedForwardBlock(d_model=256, d_ff=1024)
emb     = hb.EmbeddingBlock(vocab_size=30000, d_model=256)

# --- Attention ---
mha     = hb.MultiHeadAttentionBlock(d_model=256, num_heads=8)
self_a  = hb.SelfAttentionBlock(d_model=256, num_heads=8, causal=True)
cross_a = hb.CrossAttentionBlock(d_model=256, num_heads=8)

# --- Encoders ---
encoder = hb.TransformerEncoderBlock(d_model=256, num_heads=8, d_ff=1024, num_layers=6)

# --- Decoders ---
decoder = hb.TransformerDecoderBlock(d_model=256, num_heads=8, d_ff=1024, num_layers=6)

# --- Heads ---
cls_head   = hb.ClassificationHead(d_model=256, num_classes=10)
lm_head    = hb.LanguageModelHead(d_model=256, vocab_size=30000)
tok_head   = hb.TokenClassificationHead(d_model=256, num_classes=5)
```

## Building a Transformer from Blocks

```python
import torch
import torch.nn as nn
import haloblocks as hb


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, d_ff=1024,
                 num_enc_layers=3, num_dec_layers=3, num_classes=10):
        super().__init__()
        self.src_emb = hb.EmbeddingBlock(vocab_size, d_model)
        self.tgt_emb = hb.EmbeddingBlock(vocab_size, d_model)
        self.encoder = hb.TransformerEncoderBlock(d_model, num_heads, d_ff, num_enc_layers)
        self.decoder = hb.TransformerDecoderBlock(d_model, num_heads, d_ff, num_dec_layers)
        self.head    = hb.ClassificationHead(d_model, num_classes, pooling="mean")

    def forward(self, src, tgt):
        memory = self.encoder(self.src_emb(src))
        decoded = self.decoder(self.tgt_emb(tgt), memory)
        return self.head(decoded)


model = SimpleTransformer(vocab_size=1000)
src = torch.randint(0, 1000, (2, 10))
tgt = torch.randint(0, 1000, (2, 8))
logits = model(src, tgt)   # shape: (2, 10)
```

## Available Blocks

| Category  | Block                          | Description                                  |
|-----------|--------------------------------|----------------------------------------------|
| Base      | `Block`                        | Base class for all blocks                    |
| Layers    | `LinearBlock`                  | Linear (fully-connected) layer               |
| Layers    | `LayerNormBlock`               | Layer normalisation                          |
| Layers    | `DropoutBlock`                 | Dropout                                      |
| Layers    | `FeedForwardBlock`             | Two-layer feed-forward with GELU/ReLU        |
| Layers    | `EmbeddingBlock`               | Token embedding                              |
| Attention | `MultiHeadAttentionBlock`      | Scaled dot-product multi-head attention      |
| Attention | `SelfAttentionBlock`           | Self-attention (optionally causal)           |
| Attention | `CrossAttentionBlock`          | Cross-attention between two sequences        |
| Encoders  | `TransformerEncoderLayerBlock` | Single Transformer encoder layer            |
| Encoders  | `TransformerEncoderBlock`      | Stack of Transformer encoder layers         |
| Decoders  | `TransformerDecoderLayerBlock` | Single Transformer decoder layer            |
| Decoders  | `TransformerDecoderBlock`      | Stack of Transformer decoder layers         |
| Heads     | `ClassificationHead`           | Sequence-level classification               |
| Heads     | `LanguageModelHead`            | Token-level vocabulary projection           |
| Heads     | `TokenClassificationHead`      | Per-token classification (NER, POS, etc.)   |

## Running Tests

```bash
pip install pytest
pytest
```

