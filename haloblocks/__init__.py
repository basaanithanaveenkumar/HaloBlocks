"""HaloBlocks: a Python library where all model components are treated as blocks.

Blocks are composable PyTorch modules covering:
- Base: Block
- Layers: LinearBlock, LayerNormBlock, DropoutBlock, FeedForwardBlock, EmbeddingBlock
- Attention: MultiHeadAttentionBlock, SelfAttentionBlock, CrossAttentionBlock
- Heads: ClassificationHead, LanguageModelHead, TokenClassificationHead
- Encoders: TransformerEncoderLayerBlock, TransformerEncoderBlock
- Decoders: TransformerDecoderLayerBlock, TransformerDecoderBlock
"""

from .base import Block
from .layers import (
    LinearBlock,
    LayerNormBlock,
    DropoutBlock,
    FeedForwardBlock,
    EmbeddingBlock,
)
from .attention import (
    MultiHeadAttentionBlock,
    SelfAttentionBlock,
    CrossAttentionBlock,
)
from .heads import (
    ClassificationHead,
    LanguageModelHead,
    TokenClassificationHead,
)
from .encoders import TransformerEncoderLayerBlock, TransformerEncoderBlock
from .decoders import TransformerDecoderLayerBlock, TransformerDecoderBlock

__all__ = [
    "Block",
    # Layers
    "LinearBlock",
    "LayerNormBlock",
    "DropoutBlock",
    "FeedForwardBlock",
    "EmbeddingBlock",
    # Attention
    "MultiHeadAttentionBlock",
    "SelfAttentionBlock",
    "CrossAttentionBlock",
    # Heads
    "ClassificationHead",
    "LanguageModelHead",
    "TokenClassificationHead",
    # Encoders
    "TransformerEncoderLayerBlock",
    "TransformerEncoderBlock",
    # Decoders
    "TransformerDecoderLayerBlock",
    "TransformerDecoderBlock",
]

__version__ = "0.1.0"
