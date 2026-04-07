"""Attention building blocks.

Import classes directly::

    from haloblocks.blocks.attention import MultiQueryAttention

Or use the package as a namespace::

    from haloblocks.blocks import attention
    layer = attention.MultiQueryAttention(emb_dim=256, num_heads=8)

Submodules (e.g. ``multi_query_attention``) remain available for granular imports.
"""

from . import (
    cross_attention,
    gated_attention,
    grouped_query_attention,
    linear_attention,
    masking,
    multi_head_latent_attention,
    multi_query_attention,
    scaled_dot_product_attention,
    self_attention,
    sliding_window_attention,
    trinity_attention,
)
from .cross_attention import CrossAttention, HeadCrossAttention, MultiHeadCrossAttention
from .gated_attention import GatedAttention, GatedAttentionWithMask, GatedCrossAttention
from .grouped_query_attention import GroupedQueryAttention
from .linear_attention import (
    ELUFeatureMap,
    ExpFeatureMap,
    FeatureMap,
    LinearAttention,
    LinearAttentionWithRoPE,
    LinearCrossAttention,
    ReLUFeatureMap,
)
from .multi_head_latent_attention import MultiHeadLatentAttention
from .multi_query_attention import MultiQueryAttention
from .scaled_dot_product_attention import ScaledDotProductAttention
from .self_attention import HeadAttention, MultiHeadAttention, SelfAttention
from .sliding_window_attention import (
    DilatedSlidingWindowAttention,
    DynamicSlidingWindowAttention,
    SlidingWindowAttention,
)
from .trinity_attention import TrinityAttention, TrinityCrossAttention

__all__ = [
    "cross_attention",
    "gated_attention",
    "grouped_query_attention",
    "linear_attention",
    "multi_head_latent_attention",
    "multi_query_attention",
    "scaled_dot_product_attention",
    "self_attention",
    "sliding_window_attention",
    "trinity_attention",
    "masking",
    "CrossAttention",
    "DilatedSlidingWindowAttention",
    "DynamicSlidingWindowAttention",
    "ELUFeatureMap",
    "ExpFeatureMap",
    "FeatureMap",
    "GatedAttention",
    "GatedAttentionWithMask",
    "GatedCrossAttention",
    "GroupedQueryAttention",
    "HeadAttention",
    "HeadCrossAttention",
    "LinearAttention",
    "LinearAttentionWithRoPE",
    "LinearCrossAttention",
    "MultiHeadAttention",
    "MultiHeadCrossAttention",
    "MultiHeadLatentAttention",
    "MultiQueryAttention",
    "ReLUFeatureMap",
    "ScaledDotProductAttention",
    "SelfAttention",
    "SlidingWindowAttention",
    "TrinityAttention",
    "TrinityCrossAttention",
]
