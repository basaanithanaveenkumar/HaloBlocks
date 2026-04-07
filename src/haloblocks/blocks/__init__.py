"""Registered block implementations (one subpackage per family).

- ``attention`` — attention layers (self-, cross-, GQA, linear, etc.).
- ``norm`` — RMSNorm and other norm layers.
- ``mlp`` — feed-forward networks (often called "FFN" in papers).
- ``moe`` — mixture-of-experts modules.
- ``positional_embedding`` — sinusoidal, learned, RoPE, ALiBi, …
- ``transformer`` — transformer block and decoder wrappers.
- ``vla`` — VLA / flow-matching decoder utilities.

The empty ``heads`` placeholder was removed; multi-head logic lives under ``attention``.

**Registry access:** Any ``@BlockRegistry.register()``-ed class is also available as
``blocks.ClassName`` (e.g. ``blocks.MultiHeadAttention``), in addition to subpackages like
``blocks.attention``. Import with ``from haloblocks import blocks``.
"""

from __future__ import annotations

import sys

from . import attention
from . import mlp as mlp_pkg
from . import moe, norm, positional_embedding, transformer, vla

# Importing the ``mlp`` subpackage also binds ``blocks.mlp`` to that package module,
# which would shadow the registry key ``mlp``. Keep only ``mlp_pkg`` for the package.
_sys = sys.modules[__name__]
_sys.__dict__.pop("mlp", None)

__all__ = [
    "attention",
    "norm",
    "positional_embedding",
    "moe",
    "transformer",
    "vla",
    "mlp_pkg",
]


def __getattr__(name: str):
    from haloblocks.core.registry import BlockRegistry

    try:
        return BlockRegistry.get(name)
    except KeyError as e:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from e


def __dir__():
    from haloblocks.core.registry import BlockRegistry

    base = set(globals().keys()) | set(__all__) | set(BlockRegistry._registry.keys())
    return sorted(base)
