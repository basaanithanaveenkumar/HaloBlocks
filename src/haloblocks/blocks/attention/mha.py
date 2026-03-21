import torch
import torch.nn as nn
import torch.nn.functional as F

from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry
from .utils import RMSNorm

@BlockRegistry.register("multi_head_latent_attention")
class MultiHeadLatentAttention(Block):
    """
    Multi-Head Latent Attention with absorption trick.

    This block compresses keys and values into a low‑dimensional latent space,
    then uses an absorption trick to avoid expanding the keys during attention
    computation. The absorption trick combines the query projection with the
    key up‑projection, resulting in significant memory savings during inference.

    The implementation follows the principles of DeepSeek‑V2, where a shared
    down‑projection is used for keys and values, followed by separate head‑wise
    up‑projections. The absorption trick is applied by projecting queries into
    the latent key space using the key up‑projection weights.

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        latent_dim (int, optional): Dimension of the compressed latent space.
            If None, defaults to emb_dim // 4.
        dropout (float): Dropout probability applied to attention weights.
        use_q_norm (bool): If True, applies RMSNorm to queries after splitting heads.
        use_k_norm (bool): If True, applies RMSNorm to the compressed keys.
        tie_kv_down (bool): If True, shares the down‑projection for keys and values.
            (Not implemented – kept for future extension.)
    """
    def __init__(
        self,
        emb_dim=256,
        num_heads=8,
        latent_dim=None,
        dropout=0.0,
        use_q_norm=False,
        use_k_norm=False,
        tie_kv_down=False,          # reserved for future use
    ):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.latent_dim = latent_dim if latent_dim is not None else emb_dim // 4

        # Down‑projections for keys and values (separate by default)
        self.w_k_down = nn.Linear(emb_dim, self.latent_dim, bias=False)
        self.w_v_down = nn.Linear(emb_dim, self.latent_dim, bias=False)

        # Head‑wise up‑projections for keys and values
        # Shape: (num_heads, head_dim, latent_dim)
        self.w_k_up = nn.Parameter(torch.empty(num_heads, self.head_dim, self.latent_dim))
        self.w_v_up = nn.Parameter(torch.empty(num_heads, self.head_dim, self.latent_dim))

        # Query projection (full dimension, later split into heads)
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)

        # Output projection
        self.wo = nn.Linear(emb_dim, emb_dim, bias=False)

        # Optional normalisations
        self.q_norm = RMSNorm(self.head_dim) if use_q_norm else None
        self.k_norm = RMSNorm(self.latent_dim) if use_k_norm else None

        self.dropout_layer = nn.Dropout(dropout)

        # Initialise parameters
        self._init_weights()

    def _init_weights(self):
        """Initialise the learnable weight matrices."""
        nn.init.xavier_uniform_(self.w_k_up)
        nn.init.xavier_uniform_(self.w_v_up)
        # Linear layers are initialised with default PyTorch init (Kaiming uniform),
        # which is fine for typical transformers.

    def forward(self, query, key=None, value=None, causal_mask=None, **kwargs):
        """
        Compute multi‑head latent attention with absorption.

        Args:
            query (torch.Tensor): Query tensor of shape (batch, tgt_len, emb_dim).
            key (torch.Tensor, optional): Key tensor. Defaults to query.
            value (torch.Tensor, optional): Value tensor. Defaults to key.
            causal_mask (torch.Tensor, optional): Boolean mask of shape
                (tgt_len, src_len) or broadcastable to
                (batch, 1, tgt_len, src_len). Positions with `0` are masked.
            **kwargs: Ignored.

        Returns:
            torch.Tensor: Output tensor of shape (batch, tgt_len, emb_dim).
        """
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size = query.size(0)
        tgt_len = query.size(1)
        src_len = key.size(1)

        # ----- compress keys and values -----
        k_c = self.w_k_down(key)          # (batch, src_len, latent_dim)
        v_c = self.w_v_down(value)        # (batch, src_len, latent_dim)

        # ----- query projection and head split -----
        q = self.w_q(query)                # (batch, tgt_len, emb_dim)
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)              # (batch, num_heads, tgt_len, head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)

        # optional norm on compressed keys
        if self.k_norm is not None:
            k_c = self.k_norm(k_c)

        # ----- absorption trick: project queries into latent key space -----
        # q_abs = einsum('b h t d, h d l -> b h t l', q, w_k_up)
        q_abs = torch.einsum('b h t d, h d l -> b h t l', q, self.w_k_up)
        # q_abs shape: (batch, num_heads, tgt_len, latent_dim)

        # ----- attention scores -----
        # Expand k_c to match the head dimension: (batch, 1, src_len, latent_dim)
        k_c_expanded = k_c.unsqueeze(1)
        scores = torch.matmul(q_abs, k_c_expanded.transpose(-2, -1))   # (batch, num_heads, tgt_len, src_len)
        scores = scores / (self.head_dim ** 0.5)

        if causal_mask is not None:
            # Broadcast mask to (batch, 1, tgt_len, src_len) if needed
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # ----- weighted sum of compressed values -----
        # v_c_expanded: (batch, 1, src_len, latent_dim) – broadcast over heads
        v_c_expanded = v_c.unsqueeze(1)
        context_latent = torch.matmul(attn_weights, v_c_expanded)   # (batch, num_heads, tgt_len, latent_dim)

        # ----- up‑project values back to head dimension -----
        # context = einsum('b h t l, h d l -> b h t d', context_latent, w_v_up)
        context = torch.einsum('b h t l, h d l -> b h t d', context_latent, self.w_v_up)
        # context shape: (batch, num_heads, tgt_len, head_dim)

        # ----- merge heads and final output projection -----
        context = context.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.emb_dim)
        output = self.wo(context)

        return output