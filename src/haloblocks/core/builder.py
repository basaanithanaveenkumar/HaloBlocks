"""
Composable Transformer Block Builder.

Lets users assemble a transformer layer from *any* registered HaloBlocks
components rather than being locked to a single attention + FFN pair.
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from haloblocks.blocks.norm import RMSNorm
from haloblocks.core.block import Block
from haloblocks.core.factory import BlockFactory
from haloblocks.core.registry import BlockRegistry

# ──────────────────── helpers ────────────────────


def _build_norm(kind: str, dim: int) -> nn.Module:
    """Return a normalisation layer by name."""
    kind = kind.lower()
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    elif kind == "rmsnorm":
        return RMSNorm(dim, eps=1e-6)
    else:
        raise ValueError(f"Unknown norm type: {kind!r}. Use 'layernorm' or 'rmsnorm'.")


def _resolve_block(
    spec: Union[None, str, dict, Block],
    emb_dim: int,
    default_type: str,
    default_kwargs: Optional[Dict[str, Any]] = None,
) -> Block:
    """
    Resolve a user-provided block specification into a ``Block`` instance.

    *spec* can be:
    - ``None``       → create *default_type* with *default_kwargs*.
    - a ``str``      → treat as registry key, merge with *default_kwargs*.
    - a ``dict``     → must contain ``"type"``; passed to ``BlockFactory.create``.
    - a ``Block``    → used as-is.
    """
    import inspect

    def _build_kwargs(registry_key: str, extra: Optional[dict]) -> dict:
        """Build kwargs, only injecting emb_dim if the target class accepts it."""
        cls = BlockRegistry.get(registry_key)
        sig = inspect.signature(cls.__init__)
        params = sig.parameters

        kwargs = {}
        if "emb_dim" in params:
            kwargs["emb_dim"] = emb_dim
        if extra:
            kwargs.update(extra)
        return kwargs

    if spec is None:
        return BlockFactory.create(default_type, **_build_kwargs(default_type, default_kwargs))

    if isinstance(spec, Block):
        return spec

    if isinstance(spec, str):
        return BlockFactory.create(spec, **_build_kwargs(spec, default_kwargs))

    if isinstance(spec, dict):
        cfg = spec.copy()
        registry_key = cfg.get("type", default_type)
        cls = BlockRegistry.get(registry_key)
        sig = inspect.signature(cls.__init__)
        if "emb_dim" in sig.parameters:
            cfg.setdefault("emb_dim", emb_dim)
        return BlockFactory.create(cfg)

    raise TypeError(
        f"Cannot resolve block spec of type {type(spec).__name__!r}. " "Expected None, str, dict, or Block instance."
    )


# ──────────────────── operation-order presets ────────────────────

_COMPUTE_OPS = frozenset({"self_attn", "cross_attn", "ffn"})
_VALID_OPS = _COMPUTE_OPS | {"norm"}

PRE_NORM: Tuple[Tuple[str, ...], ...] = (("norm", "self_attn"), ("norm", "ffn"))
POST_NORM: Tuple[Tuple[str, ...], ...] = (("self_attn", "norm"), ("ffn", "norm"))
PRE_NORM_CROSS: Tuple[Tuple[str, ...], ...] = (
    ("norm", "self_attn"),
    ("norm", "cross_attn"),
    ("norm", "ffn"),
)
POST_NORM_CROSS: Tuple[Tuple[str, ...], ...] = (
    ("self_attn", "norm"),
    ("cross_attn", "norm"),
    ("ffn", "norm"),
)
SANDWICH_NORM: Tuple[Tuple[str, ...], ...] = (
    ("norm", "self_attn", "norm"),
    ("norm", "ffn", "norm"),
)


def _coerce_operation_order(
    order: Sequence,
) -> Tuple[Tuple[str, ...], ...]:
    """Normalise *operation_order* to a canonical tuple-of-tuples.

    Accepts two formats:

    * **Nested** (recommended) — each inner tuple is one sublayer::

          (('norm', 'self_attn'), ('norm', 'ffn'))

    * **Flat** (convenience) — auto-grouped by splitting before each
      compute op::

          ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
          # → (('self_attn', 'norm'), ('cross_attn', 'norm'), ('ffn', 'norm'))
    """
    if not order:
        raise ValueError("operation_order must not be empty")

    # Already nested?
    if isinstance(order[0], (tuple, list)):
        for sub in order:
            for op in sub:
                if op not in _VALID_OPS:
                    raise ValueError(f"Unknown op {op!r}; expected one of {sorted(_VALID_OPS)}")
        return tuple(tuple(s) for s in order)

    # Flat → split before each compute op (after the first one encountered).
    for op in order:
        if op not in _VALID_OPS:
            raise ValueError(f"Unknown op {op!r}; expected one of {sorted(_VALID_OPS)}")

    sublayers: list = []
    current: list = []
    for op in order:
        if op in _COMPUTE_OPS and current and any(o in _COMPUTE_OPS for o in current):
            sublayers.append(tuple(current))
            current = []
        current.append(op)
    if current:
        sublayers.append(tuple(current))
    return tuple(sublayers)


# ──────────────────── builder ────────────────────


@BlockRegistry.register()
class TransformerBlockBuilder(Block):
    """
    A composable transformer layer built from user-selected components.

    Each *slot* (self-attention, cross-attention, feed-forward) can be:

    - a registry key string  (e.g. ``"GroupedQueryAttention"``)
    - a config dict          (e.g. ``{"type": "SlidingWindowAttention", ...}``)
    - a pre-built ``Block``  instance

    **Operation order** controls the sequence of operations and where
    normalisation / residual connections are applied.  Each inner tuple is
    one *sublayer*.  Within a sublayer the residual is saved at the start
    and added back immediately after the compute op; any norms before the
    compute act as pre-norms, any after act as post-norms::

        # PRE_NORM (default)
        x → norm → self_attn → +residual → norm → ffn → +residual → out

        # POST_NORM
        x → self_attn → +residual → norm → ffn → +residual → norm → out

        # POST_NORM_CROSS (encoder-decoder)
        x → self_attn → +residual → norm
          → cross_attn(ctx) → +residual → norm
          → ffn → +residual → norm → out

    Named presets (class constants and module-level):

    ==================== =========================================================
    ``PRE_NORM``         ``(('norm','self_attn'), ('norm','ffn'))``
    ``POST_NORM``        ``(('self_attn','norm'), ('ffn','norm'))``
    ``PRE_NORM_CROSS``   pre-norm with a cross-attention sublayer
    ``POST_NORM_CROSS``  post-norm with a cross-attention sublayer
    ``SANDWICH_NORM``    ``(('norm','self_attn','norm'), ('norm','ffn','norm'))``
    ==================== =========================================================

    A flat tuple is also accepted and auto-split at each compute op::

        ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        # → equivalent to POST_NORM_CROSS

    Args:
        emb_dim (int): Embedding / model dimension.
        attn: Self-attention block spec.  Defaults to ``MultiHeadAttention``
            with ``num_heads=8``.
        attn_kwargs: Extra kwargs when *attn* is a string key.
        cross_attn: Cross-attention block spec.  Defaults to
            ``MultiHeadCrossAttention`` (created only when
            ``'cross_attn'`` appears in *operation_order*).
        cross_attn_kwargs: Extra kwargs when *cross_attn* is a string key.
        ffn: Feed-forward block spec.  Defaults to a 2-layer GELU MLP
            with ``4×`` expansion.
        ffn_kwargs: Extra kwargs when *ffn* is a string key.
        norm (str): ``"layernorm"`` (default) or ``"rmsnorm"``.
        drop_fact (float): Dropout on residual paths.  ``0.0`` by default.
        operation_order: Sequence of sublayer tuples.  Defaults to
            :data:`PRE_NORM`.

    Example::

        # GQA + cross-attention + MoE, post-norm
        block = hb.create(
            'TransformerBlockBuilder',
            emb_dim=512,
            attn={'type': 'GroupedQueryAttention', 'num_heads': 8,
                  'num_kv_heads': 2},
            cross_attn='MultiHeadCrossAttention',
            ffn={'type': 'DeepseekMoE', 'hid_dim': 1024,
                 'num_router_exprts': 8, 'best_k': 2,
                 'num_shared_exprts': 1},
            norm='rmsnorm',
            drop_fact=0.1,
            operation_order=TransformerBlockBuilder.POST_NORM_CROSS,
        )
    """

    # ── named presets ──
    PRE_NORM = PRE_NORM
    POST_NORM = POST_NORM
    PRE_NORM_CROSS = PRE_NORM_CROSS
    POST_NORM_CROSS = POST_NORM_CROSS
    SANDWICH_NORM = SANDWICH_NORM

    def __init__(
        self,
        emb_dim: int = 256,
        attn: Union[None, str, dict, Block] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        cross_attn: Union[None, str, dict, Block] = None,
        cross_attn_kwargs: Optional[Dict[str, Any]] = None,
        ffn: Union[None, str, dict, Block] = None,
        ffn_kwargs: Optional[Dict[str, Any]] = None,
        norm: str = "layernorm",
        drop_fact: float = 0.0,
        operation_order: Optional[tuple] = None,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self._norm_type = norm

        # ── parse operation order ──
        parsed = _coerce_operation_order(operation_order or PRE_NORM)

        has_cross = any("cross_attn" in sub for sub in parsed)
        norm_count = sum(op == "norm" for sub in parsed for op in sub)

        self._sublayer_plans: list = []
        norm_idx = 0
        for sublayer_ops in parsed:
            plan: list = []
            for op in sublayer_ops:
                if op == "norm":
                    plan.append(("norm", norm_idx))
                    norm_idx += 1
                else:
                    plan.append((op,))
            self._sublayer_plans.append(plan)

        # ── norms (one per 'norm' token in the order) ──
        self.norms = nn.ModuleList([_build_norm(norm, emb_dim) for _ in range(norm_count)])

        # ── self-attention ──
        default_attn_kw: dict = {"num_heads": 8}
        if attn_kwargs:
            default_attn_kw.update(attn_kwargs)
        self.attn = _resolve_block(attn, emb_dim, "MultiHeadAttention", default_attn_kw)

        # ── cross-attention (only when the order uses it) ──
        self._has_cross = has_cross
        if has_cross:
            default_cross_kw: dict = {"num_heads": 8}
            if cross_attn_kwargs:
                default_cross_kw.update(cross_attn_kwargs)
            self.cross_attn_block = _resolve_block(cross_attn, emb_dim, "MultiHeadCrossAttention", default_cross_kw)

        # ── feed-forward ──
        default_ffn_kw: dict = {
            "input_dim": emb_dim,
            "hidden_dims": [emb_dim * 4],
            "output_dim": emb_dim,
            "activation": "gelu",
        }
        if ffn_kwargs:
            default_ffn_kw.update(ffn_kwargs)
        self.ffn = _resolve_block(ffn, emb_dim, "MLP", default_ffn_kw)

        # ── residual dropout (one per compute type) ──
        def _make_drop() -> nn.Module:
            return nn.Dropout(drop_fact) if drop_fact > 0 else nn.Identity()

        self.drop_attn = _make_drop()
        if has_cross:
            self.drop_cross = _make_drop()
        self.drop_ffn = _make_drop()

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the configured operation order.

        Args:
            x: Input tensor ``(batch, seq_len, emb_dim)``.
            context: Context tensor for cross-attention
                ``(batch, ctx_len, emb_dim)``.  Required when
                ``'cross_attn'`` appears in the operation order.
            **kwargs: Forwarded to attention blocks (e.g. ``mask``).

        Returns:
            Tensor with the same shape as *x*.
        """
        for plan in self._sublayer_plans:
            residual = x
            for step in plan:
                op = step[0]
                if op == "norm":
                    x = self.norms[step[1]](x)
                elif op == "self_attn":
                    x = self.attn(x, **kwargs)
                    x = residual + self.drop_attn(x)
                elif op == "cross_attn":
                    if context is None:
                        raise ValueError(
                            "operation_order includes 'cross_attn' but no " "'context' tensor was passed to forward()."
                        )
                    x = self.cross_attn_block(x, context, **kwargs)
                    x = residual + self.drop_cross(x)
                elif op == "ffn":
                    x = self.ffn(x)
                    x = residual + self.drop_ffn(x)
        return x

    # ── convenience: stack N identical layers ─────────────────────────

    def stack(self, num_layers: int) -> "StackedTransformerBlocks":
        """
        Return a ``StackedTransformerBlocks`` with *num_layers* copies of this
        builder's configuration (independent weights) plus a final norm.
        """
        import copy

        layers = nn.ModuleList([copy.deepcopy(self) for _ in range(num_layers)])
        return StackedTransformerBlocks(layers, self.emb_dim, _build_norm(self._norm_type, self.emb_dim))


@BlockRegistry.register()
class StackedTransformerBlocks(Block):
    """
    A stack of ``TransformerBlockBuilder`` layers with a final norm.

    This is the output of :meth:`TransformerBlockBuilder.stack`.

    Args:
        layers (nn.ModuleList): Transformer block layers.
        emb_dim (int): Embedding dimension.
        norm_template (nn.Module): Norm instance to deep-copy for the final norm.
    """

    def __init__(self, layers: nn.ModuleList, emb_dim: int, norm_template: nn.Module):
        super().__init__()
        self.layers = layers
        import copy

        self.final_norm = copy.deepcopy(norm_template)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, **kwargs)
        return self.final_norm(x)


@BlockRegistry.register()
class StackedTransformerBlock(Block):
    """
    Multi-layer transformer stack, constructible from a config dict.

    Equivalent to ``TransformerBlockBuilder(...).stack(num_layers)``.

    Accepts the same arguments as :class:`TransformerBlockBuilder` plus
    ``num_layers``.
    """

    def __init__(
        self,
        emb_dim: int = 256,
        num_layers: int = 1,
        attn: Union[None, str, dict, Block] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        cross_attn: Union[None, str, dict, Block] = None,
        cross_attn_kwargs: Optional[Dict[str, Any]] = None,
        ffn: Union[None, str, dict, Block] = None,
        ffn_kwargs: Optional[Dict[str, Any]] = None,
        norm: str = "layernorm",
        drop_fact: float = 0.0,
        operation_order: Optional[tuple] = None,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        base = TransformerBlockBuilder(
            emb_dim=emb_dim,
            attn=attn,
            attn_kwargs=attn_kwargs,
            cross_attn=cross_attn,
            cross_attn_kwargs=cross_attn_kwargs,
            ffn=ffn,
            ffn_kwargs=ffn_kwargs,
            norm=norm,
            drop_fact=drop_fact,
            operation_order=operation_order,
        )
        self.stacked = base.stack(num_layers)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.stacked(x, **kwargs)
