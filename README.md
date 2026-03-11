<p align="center">
  <img src="assets/logo.png" width="200" alt="HaloBlocks Logo">
</p>

# HaloBlocks

A modern, high-performance Python library for modular neural network components. In **HaloBlocks**, every model component is treated as a first-class, composable "block"—from specialized attention layers and heads to advanced Mixture-of-Experts (MoE) modules.

## Features

- **Modular Architecture**: Build complex models by composing small, reusable blocks.
- **Mixture of Experts**: Built-in support for Deepseek-style MoE with routed and shared experts.
- **VLA Integration**: Includes specialized blocks for Vision-Language-Action models, such as Flow Matching action decoders.
- **Aesthetic First**: Designed for clarity, performance, and ease of use.

## Installation

```bash
pip install haloblocks
```

## Quick Start

```python
from haloblocks.core.factory import BlockFactory

# Create a Multi-Head Attention block
config = {
    'type': 'multi_head_attn',
    'emb_dim': 256,
    'num_heads': 8
}
attn_block = BlockFactory.create(config)
```
