---
name: haloblocks-compose
description: Build models from HaloBlocks — pick attention/FFN/positional blocks, assemble layers with TransformerBlockBuilder (operation_order, norm, cross-attention), stack them, and express whole models as config dicts/YAML. Use when a user wants a model or experiment assembled from HaloBlocks rather than a new block.
---

# Composing models

## Three equivalent APIs

```python
from haloblocks import blocks; import haloblocks as hb
blocks.GroupedQueryAttention(emb_dim=512, num_heads=8, num_kv_heads=2)
hb.create("GroupedQueryAttention", emb_dim=512, num_heads=8, num_kv_heads=2)
hb.create({"type": "GroupedQueryAttention", "emb_dim": 512, "num_heads": 8, "num_kv_heads": 2})
```

## One layer, any parts

```python
layer = hb.create(
    "TransformerBlockBuilder",
    emb_dim=256,
    attn={"type": "GroupedQueryAttention", "num_heads": 8, "num_kv_heads": 2},
    ffn={"type": "DeepseekMoE", "emb_dim": 256, "hid_dim": 512,
         "num_router_exprts": 8, "best_k": 2, "num_shared_exprts": 1},
    norm="rmsnorm",
)                                                   # 3.71M params; (2,16,256) -> (2,16,256)
```

Use the **dict** form for `ffn`/`attn` when passing block-specific kwargs. The string form
plus `ffn_kwargs` currently fails for `DeepseekMoE` (see `haloblocks-dev`).

`operation_order` controls pre-/post-norm and cross-attention placement:

```python
# constants in haloblocks.core.builder
PRE_NORM  = (("norm", "self_attn"), ("norm", "ffn"))                 # default
POST_NORM = (("self_attn", "norm"), ("ffn", "norm"))
PRE_NORM_CROSS = (("norm", "self_attn"), ("norm", "cross_attn"), ("norm", "ffn"))   # needs cross_attn=...; POST_NORM_CROSS also exists
```

A flat list such as `("self_attn", "norm", "ffn", "norm")` is auto-grouped.

## Whole model as config

```python
cfg = {"type": "CompositeBlock", "blocks": [
    {"type": "SinusoidalPositionalEmbedding", "emb_dim": 256, "max_len": 128},
    {"type": "StackedTransformerBlock", "emb_dim": 256, "num_layers": 4,
     "attn": {"type": "TrinityAttention", "num_heads": 8}, "norm": "rmsnorm"},
]}
model = hb.create(cfg)          # 3.42M params
```

`CompositeBlock` forwards the same `**kwargs` (e.g. `mask`) to every child, so every block in
a pipeline must accept `**kwargs`.

## Choosing parts

| Need | Block |
|---|---|
| Cheap KV cache | `MultiQueryAttention`, `GroupedQueryAttention`, `MultiHeadLatentAttention` (absorbed low-rank KV) |
| Long sequences | `SlidingWindowAttention` (and Dilated/Dynamic), `LinearAttention` (ELU/ReLU/Exp feature maps, optional RoPE) |
| Stable training / attention sinks | `GatedAttention` / `TrinityAttention` (QK-norm + sigmoid output gate) |
| Capacity at fixed compute | `DeepseekMoE` |
| Positions | `RotaryPositionalEmbedding(head_dim=...)`, `AlibiPositionalBias`, sinusoidal, learned |
| Robot actions | `FlowActionDecoder` (flow matching) |
