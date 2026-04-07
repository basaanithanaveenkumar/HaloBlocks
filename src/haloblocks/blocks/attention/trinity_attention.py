import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

from ..norm import RMSNorm


@BlockRegistry.register()
class TrinityAttention(Block):
    """
    Trinity Attention mechanism.

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        use_q_norm (bool): If True, applies RMSNorm to queries. Defaults to True.
        use_k_norm (bool): If True, applies RMSNorm to keys. Defaults to True.
    """

    def __init__(self, emb_dim=256, num_heads=4, use_q_norm=True, use_k_norm=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")

        self.head_dim = emb_dim // num_heads

        self.wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wv = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wg = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wo = nn.Linear(emb_dim, emb_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim) if use_q_norm else None
        self.k_norm = RMSNorm(self.head_dim) if use_k_norm else None

    def forward(self, x, **kwargs):
        """
        Q, K, V all projected from the same tensor ``x`` (self-attention).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim).

        Returns:
            torch.Tensor: The attention output.
        """
        batch, seq_len, emb_dim = x.shape
        h = self.num_heads
        head_dim = self.head_dim

        query = self.wq(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)
        key = self.wk(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)
        value = self.wv(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)

        if self.q_norm:
            query = self.q_norm(query)
        if self.k_norm:
            key = self.k_norm(key)

        output = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        output = F.softmax(output, dim=-1)
        output = torch.matmul(output, value)

        # gating
        gating = torch.sigmoid(self.wg(x))
        output = output.transpose(1, 2).reshape(batch, seq_len, emb_dim)
        output = gating * output

        output = self.wo(output)
        return output


@BlockRegistry.register()
class TrinityCrossAttention(Block):
    """
    Trinity Cross-Attention mechanism.

    Args:
        emb_dim (int): Dimensionality of the input embeddings and context.
        num_heads (int): Number of attention heads.
        use_q_norm (bool): If True, applies RMSNorm to queries. Defaults to True.
        use_k_norm (bool): If True, applies RMSNorm to keys. Defaults to True.
    """

    def __init__(self, emb_dim=256, num_heads=4, use_q_norm=True, use_k_norm=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")

        self.head_dim = emb_dim // num_heads

        self.wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wv = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wg = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wo = nn.Linear(emb_dim, emb_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim) if use_q_norm else None
        self.k_norm = RMSNorm(self.head_dim) if use_k_norm else None

    def forward(self, x, context, value_context=None, **kwargs):
        """
        Q from ``x``; K from ``context``; V from ``value_context`` when set, otherwise from ``context``.

        Args:
            x (torch.Tensor): Queries (batch_size, seq_len_q, emb_dim).
            context (torch.Tensor): Keys (batch_size, seq_len_kv, emb_dim).
            value_context (torch.Tensor, optional): Values; defaults to ``context``.

        Returns:
            torch.Tensor: The cross-attention output.
        """
        batch, seq_len_q, emb_dim = x.shape
        _, seq_len_kv, _ = context.shape
        v_src = context if value_context is None else value_context
        h = self.num_heads
        head_dim = self.head_dim

        query = self.wq(x).reshape(batch, seq_len_q, h, head_dim).transpose(1, 2)
        key = self.wk(context).reshape(batch, seq_len_kv, h, head_dim).transpose(1, 2)
        value = self.wv(v_src).reshape(batch, seq_len_kv, h, head_dim).transpose(1, 2)

        if self.q_norm:
            query = self.q_norm(query)
        if self.k_norm:
            key = self.k_norm(key)

        output = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        output = F.softmax(output, dim=-1)
        output = torch.matmul(output, value)

        # gating over queries
        gating = torch.sigmoid(self.wg(x))
        output = output.transpose(1, 2).reshape(batch, seq_len_q, emb_dim)
        output = gating * output

        output = self.wo(output)
        return output
