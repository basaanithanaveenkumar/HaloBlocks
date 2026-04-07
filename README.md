<p align="center">
<img src="https://raw.githubusercontent.com/basaanithanaveenkumar/HaloBlocks/main/assets/logo.png" width="220" alt="HaloBlocks Logo">
</p>

<h1 align="center">HaloBlocks</h1>

<p align="center">
<strong>Modern, Modular, and Composability-First Neural Network Components.</strong>
</p>

<p align="center">
<a href="https://github.com/basaanithanaveenkumar/HaloBlocks/actions"><img src="https://img.shields.io/github/actions/workflow/status/basaanithanaveenkumar/HaloBlocks/python-package.yml?branch=main&style=flat-square" alt="Build Status"></a>
<a href="https://pypi.org/project/haloblocks/"><img src="https://img.shields.io/pypi/v/haloblocks.svg?style=flat-square" alt="PyPI Version"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="License: MIT"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square" alt="Python Version"></a>
</p>

---

**HaloBlocks** is a high-performance Python library for assembling complex neural network architectures from simple, composable building blocks. Whether you are prototyping a new Transformer variant, experimenting with Mixture-of-Experts (MoE) scaling, or building a Vision-Language-Action (VLA) model from scratch, HaloBlocks provides the foundational "bricks" you need — without getting in your way.

> [!TIP]
> **New to HaloBlocks?** Jump straight into the interactive [Tutorial Notebook](notebooks/tutorial.ipynb) for a hands-on tour of the library.

---

## Features

| Feature | Description |
|---|---|
| **First-Class Composability** | Every component is a `Block` that can be freely nested and combined via `CompositeBlock` |
| **Rich Attention Zoo** | MHA, MQA, GQA, Cross, Gated, Sliding-Window, Linear, Trinity Attention & more |
| **MoE Ready** | DeepSeek-style Mixture-of-Experts with routed + shared experts and noisy Top-K routing |
| **Positional Embeddings** | Sinusoidal, Learned, RoPE (Rotary), and ALiBi — all plug-and-play |
| **VLA Integration** | Flow Matching decoder blocks for Vision-Language-Action pipelines |
| **Config-Driven** | Build entire model graphs from plain Python dicts or YAML/JSON configs via `BlockFactory` |
| **Keras-like API** | Access any block directly from `haloblocks.blocks` (registry keys) without touching a config |
| **PyTorch Native** | Zero magic — pure `nn.Module` subclasses built for speed and debuggability |

---

## Installation

```bash
# Using pip
pip install haloblocks

# Using uv (recommended — faster)
uv add haloblocks
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, NumPy ≥ 2.0

### Running tests

From a clone of this repository (at the repo root):

```bash
./scripts/run_all_unit_tests.sh
./scripts/run_all_unit_tests.sh -v
./scripts/run_all_unit_tests.sh tests/test_core.py::TestCore::test_composite_block
```

The script runs `uv sync --dev` and then `pytest`. With no arguments it runs the full suite under `tests/`; any extra arguments are forwarded to pytest.

### Releasing a new version

Requires [GitHub CLI](https://cli.github.com/) (`gh`) authenticated (`gh auth login`).

```bash
./scripts/open_release_pr.sh 0.2.0
```

This updates `version` in `pyproject.toml` (and `uv.lock` when `uv` is available), commits on branch `release/v0.2.0`, pushes it, and opens a PR to `main`. **After the PR is merged**, GitHub Actions creates tag `v0.2.0` and pushes it; the existing **Publish to PyPI** workflow runs on that tag.

---

## Quick Start

HaloBlocks supports three usage styles — choose whichever fits your workflow.

### Style 1 — Keras-like direct access

No config dictionaries needed. Import `blocks` from `haloblocks` and call any block by its class name, or import attention classes from the `attention` package:

```python
from haloblocks import blocks
from haloblocks.blocks import attention

# Multi-Head Attention (registry key = class name)
attn = blocks.MultiHeadAttention(emb_dim=512, num_heads=8)

# Same family, explicit class (namespace style)
mqa = attention.MultiQueryAttention(emb_dim=512, num_heads=8)

# Grouped-Query Attention
gqa = blocks.GroupedQueryAttention(emb_dim=512, num_heads=8, num_kv_heads=2)

# Rotary Positional Embedding (RoPE is per-head; pass head_dim, not full emb_dim)
rope = blocks.RotaryPositionalEmbedding(head_dim=64, max_len=8192)

# DeepSeek Mixture-of-Experts
moe = blocks.DeepseekMoE(
    emb_dim=512,
    hid_dim=2048,
    num_router_exprts=8,
    best_k=2,
    num_shared_exprts=2,
)
```

### Style 2 — `hb.create` convenience function

```python
import haloblocks as hb

attn = hb.create('MultiHeadAttention', emb_dim=512, num_heads=8)
moe = hb.create(
    'DeepseekMoE',
    emb_dim=512,
    hid_dim=2048,
    num_router_exprts=8,
    best_k=2,
    num_shared_exprts=2,
)
```

### Style 3 — Config-driven (YAML / JSON friendly)

Perfect for experiment configs or hyperparameter sweeps:

```python
import haloblocks as hb

config = {
    'type': 'TransformerBlock',
    'emb_dim': 768,
    'num_heads': 12,
    'mlp_dim': 3072,
    'use_moe': False,
}
block = hb.create(config)
```

### Building a Composite Model

```python
import haloblocks as hb

model = hb.CompositeBlock([
    hb.create('SinusoidalPositionalEmbedding', emb_dim=512, max_len=1024),
    hb.create('TransformerBlock', emb_dim=512, num_heads=8),
    hb.create('TransformerBlock', emb_dim=512, num_heads=8),
])
```

---

## Block Catalogue

Registry keys now match class names exactly. Use them with `blocks.<ClassName>`, `hb.create('<ClassName>', ...)`, and `hb.create({'type': '<ClassName>', ...})`.

Every block supports **three equivalent ways to instantiate**. For example, Multi-Head Attention:

```python
# 1. blocks.<ClassName>
from haloblocks import blocks
attn = blocks.MultiHeadAttention(emb_dim=256, num_heads=8)

# 2. hb.create with keyword args
import haloblocks as hb
attn = hb.create('MultiHeadAttention', emb_dim=256, num_heads=8)

# 3. hb.create with a config dict
attn = hb.create({'type': 'MultiHeadAttention', 'emb_dim': 256, 'num_heads': 8})
```

### Attention

| Block | Registry Key | Description |
|---|---|---|
| Scaled Dot-Product | `ScaledDotProductAttention` | Bare-metal scaled dot-product attention |
| Self-Attention | `SelfAttention` | Single-head self-attention (QKV from same input) |
| Attention head (sub-block) | `HeadAttention` | Single head in a reduced subspace (building block inside MHA) |
| Multi-Head Attention | `MultiHeadAttention` | Classic MHA (Vaswani et al.) |
| Multi-Query Attention | `MultiQueryAttention` | MQA — single shared KV head |
| Grouped-Query Attention | `GroupedQueryAttention` | GQA — configurable KV head groups |
| Cross-Attention (single-head) | `CrossAttention` | Single-head encoder–decoder cross-attention |
| Cross-Attention (multi-head) | `MultiHeadCrossAttention` | Multi-head cross-attention |
| Gated Attention | `GatedAttention` | Attention with gating mechanism |
| Sliding Window Attention | `SlidingWindowAttention` | Local context window (Longformer-style) |
| Linear Attention | `LinearAttention` | Sub-quadratic linear attention |
| Multi-Head Latent Attention | `MultiHeadLatentAttention` | Latent-space compressed attention |
| Trinity Attention | `TrinityAttention` | Combined local + global + linear attention |

### Positional Embeddings

| Block | Registry Key | Description |
|---|---|---|
| Sinusoidal | `SinusoidalPositionalEmbedding` | Fixed sinusoidal PE (original Transformer) |
| Learned | `LearnedPositionalEmbedding` | Trainable position embeddings |
| Rotary (RoPE) | `RotaryPositionalEmbedding` | Rotary position encoding |
| ALiBi | `AlibiPositionalBias` | Attention with Linear Biases |

### MLP

| Block | Registry Key | Description |
|---|---|---|
| MLP | `MLP` | Configurable feed-forward block (activations, bias, last-layer options) |

### Mixture-of-Experts

| Block | Registry Key | Description |
|---|---|---|
| DeepSeek MoE | `DeepseekMoE` | Routed + shared experts with noisy Top-K routing |

### Transformer

| Block | Registry Key | Description |
|---|---|---|
| Transformer Block Builder | `TransformerBlockBuilder` | Highly composable layer supporting arbitrary Attn & FFN combinations |
| Stacked Transformer (config) | `StackedTransformerBlock` | Same as `TransformerBlockBuilder` with `num_layers`; use inside `CompositeBlock` configs |
| Transformer Block | `TransformerBlock` | Pre-norm Transformer layer (attn + MLP) |
| Decoder | `DecoderTransformer` | Stacked Transformer decoder |

### Vision-Language-Action

| Block | Registry Key | Description |
|---|---|---|
| Flow Decoder | `FlowActionDecoder` | Flow-matching decoder for VLA action prediction |

---

## Project Structure

```
HaloBlocks/
├── src/haloblocks/
│  ├── __init__.py       # Top-level API: Block, BlockFactory, create
│  ├── layers.py        # Alias of ``blocks`` (same module; backward compatible)
│  ├── core/
│  │  ├── block.py      # Base Block (nn.Module subclass)
│  │  ├── registry.py     # BlockRegistry — central name → class map
│  │  ├── factory.py     # BlockFactory.create() dispatcher
│  │  ├── composite.py    # CompositeBlock for sequential pipelines
│  │  └── builder.py     # TransformerBlockBuilder, stacked blocks
│  └── blocks/
│    ├── attention/     # MHA, MQA, GQA, Cross, Gated, Sliding-Window, Linear, Trinity
│    ├── norm/          # RMSNorm (shared by attention and builder)
│    ├── positional_embedding/ # Sinusoidal, Learned, RoPE, ALiBi
│    ├── mlp/        # Configurable MLP block
│    ├── moe/        # DeepSeek Mixture-of-Experts
│    ├── transformer/    # Transformer Block & Decoder
│    └── vla/        # Flow Matching Decoder
├── scripts/
│  ├── run_all_unit_tests.sh  # uv sync --dev && pytest (optional args)
│  ├── open_release_pr.sh     # bump version + PR; merge triggers tag + PyPI
│  └── format.sh             # black, isort, pyflakes
├── notebooks/
│  └── tutorial.ipynb     # Interactive getting-started guide
├── tests/           # Pytest test suite
└── pyproject.toml
```

---

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) before opening a pull request. When adding a new block:

1. Create your module under `src/haloblocks/blocks/<category>/`.
2. Subclass `Block` and decorate with `@BlockRegistry.register()` (auto-uses the class name).
3. Re-export from the category's `__init__.py`.
4. Add a test in `tests/` and a usage example in the tutorial notebook.

---

## License

HaloBlocks is released under the [MIT License](LICENSE).

---

<p align="center">
Built with by <a href="https://github.com/basaanithanaveenkumar">Naveen</a>
</p>