import torch
import torch.nn as nn
import torch.nn.functional as F

from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

from ..norm import RMSNorm


class FeatureMap(nn.Module):
    """Base class for feature maps used in linear attention."""

    def forward(self, x):
        raise NotImplementedError


@BlockRegistry.register()
class ELUFeatureMap(FeatureMap):
    """
    ELU-based feature map for linear attention.

    Uses ELU activation + 1 to ensure positivity.
    φ(x) = ELU(x) + 1

    Args:
        eps (float): Small constant for numerical stability.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.elu(x) + 1 + self.eps


@BlockRegistry.register()
class ReLUFeatureMap(FeatureMap):
    """
    ReLU-based feature map for linear attention.

    φ(x) = ReLU(x)
    """

    def forward(self, x):
        return F.relu(x)


@BlockRegistry.register()
class ExpFeatureMap(FeatureMap):
    """
    Exponential feature map for linear attention.

    φ(x) = exp(x)
    """

    def forward(self, x):
        return torch.exp(x)


@BlockRegistry.register()
class LinearAttention(Block):
    """
    Linear Attention mechanism with O(n) complexity.

    This implementation uses kernel-based linear attention that avoids
    computing the full n×n attention matrix, reducing complexity from
    O(n²) to O(n·d²).

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        feature_map (str): Type of feature map ('elu', 'relu', or 'exp').
        use_q_norm (bool): If True, applies RMSNorm to queries. Defaults to True.
        use_k_norm (bool): If True, applies RMSNorm to keys. Defaults to True.
        causal (bool): If True, applies causal masking. Defaults to False.
        eps (float): Small constant for numerical stability.
    """

    def __init__(
        self, emb_dim=256, num_heads=4, feature_map="elu", use_q_norm=True, use_k_norm=True, causal=False, eps=1e-6
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.causal = causal
        self.eps = eps

        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")

        self.head_dim = emb_dim // num_heads

        # Attention projections
        self.wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wv = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wo = nn.Linear(emb_dim, emb_dim, bias=False)

        # Feature map selection
        if feature_map == "elu":
            self.feature_map = ELUFeatureMap(eps)
        elif feature_map == "relu":
            self.feature_map = ReLUFeatureMap()
        elif feature_map == "exp":
            self.feature_map = ExpFeatureMap()
        else:
            raise ValueError(f"Unknown feature map: {feature_map}")

        # Optional normalization
        self.q_norm = RMSNorm(self.head_dim) if use_q_norm else None
        self.k_norm = RMSNorm(self.head_dim) if use_k_norm else None

    def _linear_attention(self, Q, K, V):
        """
        Compute linear attention with O(n) complexity.

        Args:
            Q: [batch, heads, seq_len, head_dim] - Queries
            K: [batch, heads, seq_len, head_dim] - Keys
            V: [batch, heads, seq_len, head_dim] - Values

        Returns:
            Output of shape [batch, heads, seq_len, head_dim]
        """
        # Apply feature maps
        Q = self.feature_map(Q)
        K = self.feature_map(K)

        if self.causal:
            # Causal linear attention (autoregressive)
            output = self._causal_linear_attention(Q, K, V)
        else:
            # Non-causal linear attention
            # Compute KV = K^T V in O(n·d²)
            KV = torch.einsum("bhnd,bhnm->bhdm", K, V)  # [batch, heads, head_dim, head_dim]

            # Compute QKV = Q @ KV
            output = torch.einsum("bhnd,bhdm->bhnm", Q, KV)  # [batch, heads, seq_len, head_dim]

            # Normalize
            normalizer = torch.einsum("bhnd,bhd->bhn", Q, K.sum(dim=2))  # [batch, heads, seq_len]
            output = output / (normalizer.unsqueeze(-1) + self.eps)

        return output

    def _causal_linear_attention(self, Q, K, V):
        """
        Compute causal linear attention with prefix sum.

        This implements autoregressive attention where each position only
        attends to previous positions.

        Args:
            Q, K, V: Same as above

        Returns:
            Causal attention output
        """
        batch, heads, seq_len, head_dim = Q.shape

        # Initialize cumulative sums
        output = torch.zeros_like(Q)
        kv_sum = torch.zeros(batch, heads, head_dim, head_dim, device=Q.device)
        k_sum = torch.zeros(batch, heads, head_dim, device=Q.device)

        # Sequential computation (still O(n) but sequential)
        for i in range(seq_len):
            # Update cumulative KV sum
            kv_sum = kv_sum + torch.einsum("bhd,bhm->bhdm", K[:, :, i], V[:, :, i])
            k_sum = k_sum + K[:, :, i]

            # Compute output for current position
            q_i = Q[:, :, i]  # [batch, heads, head_dim]
            output_i = torch.einsum("bhd,bhdm->bhm", q_i, kv_sum)  # [batch, heads, head_dim]

            # Normalize
            normalizer = torch.einsum("bhd,bhd->bh", q_i, k_sum)  # [batch, heads]
            output_i = output_i / (normalizer.unsqueeze(-1) + self.eps)

            output[:, :, i] = output_i

        return output

    def forward(self, x, *, mask=None, **kwargs):
        """
        Computes linear attention output.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim).
            mask (torch.Tensor, optional): Attention mask (not used in linear attention
                                          due to O(n) complexity, but kept for API compatibility).

        Returns:
            torch.Tensor: The attention output.
        """
        batch, seq_len, emb_dim = x.shape
        h = self.num_heads
        head_dim = self.head_dim

        # Linear projections and reshape
        query = self.wq(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)
        key = self.wk(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)
        value = self.wv(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)

        # Apply normalization
        if self.q_norm:
            query = self.q_norm(query)
        if self.k_norm:
            key = self.k_norm(key)

        # Apply linear attention
        attention_output = self._linear_attention(query, key, value)

        # Reshape and project
        attention_output = attention_output.transpose(1, 2).reshape(batch, seq_len, emb_dim)
        output = self.wo(attention_output)

        return output


@BlockRegistry.register()
class LinearCrossAttention(LinearAttention):
    """
    Linear Cross-Attention with O(n) complexity.

    Similar to LinearAttention but for cross-attention scenarios.
    """

    def forward(self, x, context, **kwargs):
        """
        Computes linear cross-attention output.

        Args:
            x (torch.Tensor): Query sequences (batch_size, seq_len_q, emb_dim).
            context (torch.Tensor): Key/Value sequences (batch_size, seq_len_kv, emb_dim).

        Returns:
            torch.Tensor: The cross-attention output.
        """
        batch, seq_len_q, emb_dim = x.shape
        _, seq_len_kv, _ = context.shape
        h = self.num_heads
        head_dim = self.head_dim

        # Projections
        query = self.wq(x).reshape(batch, seq_len_q, h, head_dim).transpose(1, 2)
        key = self.wk(context).reshape(batch, seq_len_kv, h, head_dim).transpose(1, 2)
        value = self.wv(context).reshape(batch, seq_len_kv, h, head_dim).transpose(1, 2)

        # Normalization
        if self.q_norm:
            query = self.q_norm(query)
        if self.k_norm:
            key = self.k_norm(key)

        # Apply feature maps
        query = self.feature_map(query)
        key = self.feature_map(key)

        # Compute cross-attention efficiently
        # KV = K^T V
        KV = torch.einsum("bhnd,bhnm->bhdm", key, value)

        # Q @ KV
        attention_output = torch.einsum("bhnd,bhdm->bhnm", query, KV)

        # Normalize
        normalizer = torch.einsum("bhnd,bhd->bhn", query, key.sum(dim=2))
        attention_output = attention_output / (normalizer.unsqueeze(-1) + self.eps)

        # Reshape and project
        attention_output = attention_output.transpose(1, 2).reshape(batch, seq_len_q, emb_dim)
        output = self.wo(attention_output)

        return output


@BlockRegistry.register()
class LinearAttentionWithRoPE(LinearAttention):
    """
    Linear Attention with Rotary Position Embeddings.

    Combines linear attention with rotary positional embeddings for
    better position encoding.
    """

    def __init__(
        self,
        emb_dim=256,
        num_heads=4,
        feature_map="elu",
        use_q_norm=True,
        use_k_norm=True,
        causal=False,
        eps=1e-6,
        rope_base=10000.0,
    ):
        super().__init__(emb_dim, num_heads, feature_map, use_q_norm, use_k_norm, causal, eps)
        self.rope_base = rope_base

    def _apply_rope(self, x, positions):
        """Apply rotary position embeddings."""
        batch, heads, seq_len, head_dim = x.shape

        # Create rotary embeddings
        freqs = 1.0 / (self.rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = freqs.to(x.device)

        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # Split into pairs and rotate
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)

        return rotated_x

    def forward(self, x, *, mask=None, **kwargs):
        """Forward pass with rotary position embeddings."""
        batch, seq_len, emb_dim = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)

        # Standard forward from parent class
        h = self.num_heads
        head_dim = self.head_dim

        query = self.wq(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)
        key = self.wk(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)
        value = self.wv(x).reshape(batch, seq_len, h, head_dim).transpose(1, 2)

        # Apply RoPE
        query = self._apply_rope(query, positions)
        key = self._apply_rope(key, positions)

        if self.q_norm:
            query = self.q_norm(query)
        if self.k_norm:
            key = self.k_norm(key)

        # Continue with linear attention
        attention_output = self._linear_attention(query, key, value)
        attention_output = attention_output.transpose(1, 2).reshape(batch, seq_len, emb_dim)
        output = self.wo(attention_output)

        return output
