<p align="center">
  <img src="https://raw.githubusercontent.com/basaanithanaveenkumar/HaloBlocks/main/assets/logo.png" width="250" alt="HaloBlocks Logo">
</p>

<h1 align="center">HaloBlocks</h1>

<p align="center">
  <strong>Modern, Modular, and Composability-First Neural Network Components.</strong>
</p>

<p align="center">
  <a href="https://github.com/basaanithanaveenkumar/HaloBlocks/actions"><img src="https://img.shields.io/github/actions/workflow/status/basaanithanaveenkumar/HaloBlocks/python-package.yml?branch=main" alt="Build Status"></a>
  <a href="https://pypi.org/project/haloblocks/"><img src="https://img.shields.io/pypi/v/haloblocks.svg" alt="PyPI Version"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python Version"></a>
</p>

---

**HaloBlocks** is a high-performance Python library designed for building complex neural network architectures through simple, composable blocks. Whether you are building Transformers, Mixture-of-Experts (MoE), or Vision-Language-Action (VLA) models, HaloBlocks provides the foundational "bricks" you need.

## ✨ Key Features

- 🧩 **First-Class Composability**: Every component is treated as a "block" that can be easily nested and combined.
- 🚀 **MoE Ready**: Built-in support for advanced Mixture-of-Experts architectures, including routed and shared expert systems.
- 👁️ **VLA Integration**: Optimized blocks for Vision-Language-Action models, featuring specialized decoders and attention mechanisms.
- 🛠️ **Config-Driven Architecture**: Build entire models from JSON/YAML configurations using the `BlockFactory`.
- ⚡ **Performance Optimized**: Native PyTorch implementation with a focus on speed and memory efficiency.

## 🚀 Installation

Install the library via `pip`:

```bash
pip install haloblocks
```

Or using `uv` for faster dependency management:

```bash
uv add haloblocks
```

## 🛠️ Quick Start

> [!TIP]
> **New to HaloBlocks?** Check out our interactive [Tutorial Notebook](notebooks/tutorial.ipynb) to see the library in action!

### 1. Keras-like "Direct" Style (New! ✨)

No more nested dictionaries. Access blocks directly from `haloblocks.layers`:

```python
import haloblocks.layers as layers

# Create a block as a class instance
attn = layers.multi_head_attn(emb_dim=512, num_heads=8)
```

### 2. The Convenience `create` Function

Use the top-level `create` function for easy instantiation by string name:

```python
import haloblocks as hb

# Quick creation with keyword arguments
attn = hb.create('multi_head_attn', emb_dim=512, num_heads=8)
```

### 3. Config-Driven Style

Still fully supported and perfect for YAML/JSON configurations:

```python
config = {
    'type': 'multi_head_attn',
    'emb_dim': 512,
    'num_heads': 8
}
attn = hb.create(config)
```

## 📂 Project Structure

```text
haloblocks/
├── core/               # Foundational Block, Factory, and Registry
├── blocks/             # Specialized component implementations
│   ├── attention/      # Self-Attention, Multi-Head, Scaled Dot-Product
│   ├── moe/            # Mixture-of-Experts (DeepSeek style)
│   ├── vla/            # Vision-Language-Action specific blocks
│   └── transformer/    # Transformer layers and blocks
└── heads/              # Model output heads (Classification, LM, etc.)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) to get started.

## 📄 License

HaloBlocks is released under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/basaanithanaveenkumar">Naveen</a>
</p>
