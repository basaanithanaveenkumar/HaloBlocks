import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

from ..norm import RMSNorm

# TODO " this is similar to trinity attention, kind of redundant we need to remove"


@BlockRegistry.register()
class GatedAttention(Block):
    """
    Gated Attention mechanism with explicit gating over attention output.

    This attention mechanism applies a learnable gate to the attention output,
    allowing the model to control how much information flows through the attention
    layer based on the input.

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        use_q_norm (bool): If True, applies RMSNorm to queries. Defaults to True.
        use_k_norm (bool): If True, applies RMSNorm to keys. Defaults to True.
        gate_bias (float): Initial bias for the gate. Negative values initialize
                          the gate to be more closed. Defaults to 0.0.
    """

    def __init__(self, emb_dim=256, num_heads=4, use_q_norm=True, use_k_norm=True, gate_bias=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")

        self.head_dim = emb_dim // num_heads

        # Standard attention projections
        self.wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wv = nn.Linear(emb_dim, emb_dim, bias=False)

        # Gate projection - creates gate values based on input
        self.wg = nn.Linear(emb_dim, emb_dim, bias=True)
        if gate_bias != 0.0:
            nn.init.constant_(self.wg.bias, gate_bias)

        # Output projection
        self.wo = nn.Linear(emb_dim, emb_dim, bias=False)

        # Optional normalization
        self.q_norm = RMSNorm(self.head_dim) if use_q_norm else None
        self.k_norm = RMSNorm(self.head_dim) if use_k_norm else None

    def forward(self, x, **kwargs):
        """
        Computes the gated attention output.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim).
            **kwargs: Additional arguments (e.g., mask) for future compatibility.

        Returns:
            torch.Tensor: The gated attention output.
        """
        batch, seq_len, emb_dim = x.shape
        h = self.num_heads
        head_dim = self.head_dim

        # Linear projections and reshape for multi-head attention
        query = self.wq(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)
        key = self.wk(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)
        value = self.wv(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)

        # Apply optional normalization
        if self.q_norm:
            query = self.q_norm(query)
        if self.k_norm:
            key = self.k_norm(key)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)

        # Reshape back to original dimensions
        attention_output = attention_output.transpose(1, 2).reshape(batch, seq_len, emb_dim)

        # Compute gate from input
        gate = torch.sigmoid(self.wg(x))

        # Apply gating: element-wise multiplication of gate with attention output
        gated_output = gate * attention_output

        # Final output projection
        output = self.wo(gated_output)

        return output


@BlockRegistry.register()
class GatedCrossAttention(Block):
    """
    Gated Cross-Attention mechanism with explicit gating.

    Similar to GatedAttention but for cross-attention scenarios where query and
    key/value come from different sources. The gate is computed from the query
    to control cross-attention information flow.

    Args:
        emb_dim (int): Dimensionality of the input embeddings and context.
        num_heads (int): Number of attention heads.
        use_q_norm (bool): If True, applies RMSNorm to queries. Defaults to True.
        use_k_norm (bool): If True, applies RMSNorm to keys. Defaults to True.
        gate_bias (float): Initial bias for the gate. Negative values initialize
                          the gate to be more closed. Defaults to 0.0.
    """

    def __init__(self, emb_dim=256, num_heads=4, use_q_norm=True, use_k_norm=True, gate_bias=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")

        self.head_dim = emb_dim // num_heads

        # Attention projections
        self.wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wv = nn.Linear(emb_dim, emb_dim, bias=False)

        # Gate projection - based on query input
        self.wg = nn.Linear(emb_dim, emb_dim, bias=True)
        if gate_bias != 0.0:
            nn.init.constant_(self.wg.bias, gate_bias)

        # Output projection
        self.wo = nn.Linear(emb_dim, emb_dim, bias=False)

        # Optional normalization
        self.q_norm = RMSNorm(self.head_dim) if use_q_norm else None
        self.k_norm = RMSNorm(self.head_dim) if use_k_norm else None

    def forward(self, x, context, **kwargs):
        """
        Computes the gated cross-attention output.

        Args:
            x (torch.Tensor): Query sequences (batch_size, seq_len_q, emb_dim).
            context (torch.Tensor): Key/Value sequences (batch_size, seq_len_kv, emb_dim).
            **kwargs: Additional arguments (e.g., mask) for future compatibility.

        Returns:
            torch.Tensor: The gated cross-attention output.
        """
        batch, seq_len_q, emb_dim = x.shape
        _, seq_len_kv, _ = context.shape
        h = self.num_heads
        head_dim = self.head_dim

        # Linear projections for queries from x, keys/values from context
        query = self.wq(x).reshape(batch, seq_len_q, h, head_dim).transpose(1, 2)
        key = self.wk(context).reshape(batch, seq_len_kv, h, head_dim).transpose(1, 2)
        value = self.wv(context).reshape(batch, seq_len_kv, h, head_dim).transpose(1, 2)

        # Apply optional normalization
        if self.q_norm:
            query = self.q_norm(query)
        if self.k_norm:
            key = self.k_norm(key)

        # Scaled dot-product cross-attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)

        # Reshape back to original dimensions
        attention_output = attention_output.transpose(1, 2).reshape(batch, seq_len_q, emb_dim)

        # Compute gate from query input
        gate = torch.sigmoid(self.wg(x))

        # Apply gating: element-wise multiplication of gate with cross-attention output
        gated_output = gate * attention_output

        # Final output projection
        output = self.wo(gated_output)

        return output


@BlockRegistry.register()
class GatedAttentionWithMask(GatedAttention):
    """
    Gated Attention with masking support.

    Extends GatedAttention to support attention masking, useful for causal
    attention in autoregressive models or padding masking.

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        use_q_norm (bool): If True, applies RMSNorm to queries. Defaults to True.
        use_k_norm (bool): If True, applies RMSNorm to keys. Defaults to True.
        gate_bias (float): Initial bias for the gate. Defaults to 0.0.
    """

    def __init__(self, emb_dim=256, num_heads=4, use_q_norm=True, use_k_norm=True, gate_bias=0.0):
        super().__init__(emb_dim, num_heads, use_q_norm, use_k_norm, gate_bias)

    def forward(self, x, *, mask=None, **kwargs):
        """
        Computes masked gated attention output.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim).
            mask (torch.Tensor, optional): Attention mask of shape
                                          (batch_size, 1, seq_len, seq_len) or
                                          (batch_size, seq_len, seq_len).
                                          Values of 0 indicate positions to mask.

        Returns:
            torch.Tensor: The gated attention output.
        """
        batch, seq_len, emb_dim = x.shape
        h = self.num_heads
        head_dim = self.head_dim

        # Linear projections and reshape
        query = self.wq(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)
        key = self.wk(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)
        value = self.wv(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)

        if self.q_norm:
            query = self.q_norm(query)
        if self.k_norm:
            key = self.k_norm(key)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply mask if provided
        if mask is not None:
            # Handle different mask shapes
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)

        # Reshape and apply gate
        attention_output = attention_output.transpose(1, 2).reshape(batch, seq_len, emb_dim)
        gate = torch.sigmoid(self.wg(x))
        gated_output = gate * attention_output

        output = self.wo(gated_output)

        return output
