---
name: haloblocks-dev
description: Set up, test, format and release HaloBlocks, the pip-published (`haloblocks`) library of composable PyTorch blocks (attention zoo, RoPE/ALiBi, DeepSeek MoE, transformer builder, flow-matching VLA decoder) with a registry/factory API. Use when starting work in this repo, before changing any block, or when cutting a release.
---

# HaloBlocks development

PyPI package `haloblocks` (v0.1.3), Python ≥ 3.10, MIT.

## Setup and tests

```bash
uv sync --dev
./scripts/run_all_unit_tests.sh            # uv sync --dev && pytest; 71 tests pass (2026-09)
./scripts/run_all_unit_tests.sh -v tests/test_attention.py
./scripts/format.sh                         # black, isort, pyflakes
```

CI: `.github/workflows/test.yml` (tests), `tag_release_on_merge.yml` (tags `vX.Y.Z` when a
`release/*` PR merges) and `publish.yml` (PyPI on tag).

## Release

```bash
./scripts/open_release_pr.sh 0.2.0   # bumps pyproject + uv.lock, opens PR release/v0.2.0 → main
```

Merging the PR tags and publishes. Don't push tags by hand.

## Layout

| Path | What |
|---|---|
| `core/block.py` | `Block(ABC, nn.Module)`: every component; `forward(x, **kwargs)` |
| `core/registry.py` | `BlockRegistry.register(name=None)`: key defaults to the class name |
| `core/factory.py` | `BlockFactory.create(type_or_config, **kw)`: recursive for `"blocks"` lists |
| `core/composite.py` | `CompositeBlock([...])`: sequential; kwargs forwarded to every child |
| `core/builder.py` | `TransformerBlockBuilder` / `StackedTransformerBlock(s)`: any attn/cross-attn/FFN, `operation_order`, `norm` |
| `blocks/attention/` | SDPA, Self, Head, MHA, MQA, GQA, Cross (+MH), Gated (+mask, +cross), SlidingWindow (+Dilated, +Dynamic), Linear (+RoPE, +cross; ELU/ReLU/Exp feature maps), MLA (absorption), Trinity (+cross), `masking.py` |
| `blocks/positional_embedding/` | Sinusoidal, Learned, RoPE, ALiBi |
| `blocks/norm/` | RMSNorm |
| `blocks/mlp/`, `blocks/moe/` | configurable `MLP`; `DeepseekMoE` (shared + noisy top-k routed) |
| `blocks/transformer/` | `TransformerBlock`, `DecoderTransformer` |
| `blocks/vla/` | `FlowActionDecoder` |
| `layers.py` | backwards-compatible alias of `blocks` |

37 registry keys in total (`sorted(BlockRegistry._registry)`).

## Known discrepancies (fix or document when touching them)

- README says `TrinityAttention` is "combined local + global + linear attention". The code is
  softmax attention with per-head QK RMSNorm and a **sigmoid output gate** (`sigmoid(W_g x) ⊙ attn`),
  with no mask argument. Fix whichever is wrong.
- The README build badge points to `python-package.yml`, but the workflow is `test.yml`.
- `TransformerBlockBuilder(ffn="DeepseekMoE", ffn_kwargs=...)` (string form) fails with
  `unexpected keyword argument 'input_dim'` because the default MLP kwargs leak through.
  The dict form `ffn={"type": "DeepseekMoE", "emb_dim": ..., ...}` works.
