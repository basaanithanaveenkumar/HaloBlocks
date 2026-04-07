"""Backward-compatible alias: ``haloblocks.layers`` is the same module as ``haloblocks.blocks``.

Prefer ``from haloblocks import blocks`` and ``blocks.<registry_key>``.
"""

from __future__ import annotations

import sys

import haloblocks.blocks as _blocks

sys.modules[__name__] = sys.modules[_blocks.__name__]
