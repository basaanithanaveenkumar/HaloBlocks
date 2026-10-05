---
name: haloblocks-add-block
description: Add a new block to HaloBlocks (attention variant, positional embedding, FFN/MoE, norm, VLA head) so it works through all three APIs (blocks.<Name>, hb.create('<Name>'), config dicts), with tests and docs. Use when implementing a new layer in this library.
---

# Adding a block

1. **File** under `src/haloblocks/blocks/<category>/my_block.py`:

```python
from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

@BlockRegistry.register()                 # key = "MyBlock"
class MyBlock(Block):
    def __init__(self, emb_dim=256, num_heads=4, **kw):
        super().__init__()
        if emb_dim % num_heads:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")
        ...
    def forward(self, x, mask=None, **kwargs):   # always accept **kwargs: CompositeBlock forwards everything
        ...
```

2. **Re-export** from `blocks/<category>/__init__.py` (and `__all__`) so `blocks.MyBlock` and
   `from haloblocks.blocks.<category> import MyBlock` work.
3. **Conventions**
   - First constructor argument is `emb_dim` when the block works on model width.
     `TransformerBlockBuilder` injects `emb_dim` only if the signature accepts it. RoPE takes
     `head_dim` instead.
   - Masks: validate them with `blocks/attention/masking.py::check_attention_mask_broadcasts`.
   - Raise `ValueError` for invalid shapes and hyper-parameters, with the offending values
     in the message.
   - Pure `nn.Module` code: no global state and no device assumptions.
4. **Tests** in the matching `tests/test_*.py`: output shape, a mask test (future positions
   get zero gradient when causal), parameter count, and creation through all three APIs:

```python
blocks.MyBlock(emb_dim=64, num_heads=4)
hb.create("MyBlock", emb_dim=64, num_heads=4)
hb.create({"type": "MyBlock", "emb_dim": 64, "num_heads": 4})
```

5. **Docs**: add a row to the README "Block Catalogue", the `docs/blocks.md` table and, if
   useful, a Mermaid diagram in `docs/architecture.md`. Add a cell to
   `notebooks/tutorial.ipynb`.
6. **Changelog / version**: user-visible additions go in the next release PR
   (`scripts/open_release_pr.sh`).
