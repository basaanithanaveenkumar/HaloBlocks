import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

@BlockRegistry.register("self_attention_basic")
class SelfAttn(Block):
    """
    A basic Single-Head Self-Attention block.

    This block implements the standard scaled dot-product self-attention mechanism
    where queries, keys, and values are projected from the same input.

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        return_attn_weights (bool): If True, returns both the output and the
            attention weights. Defaults to False.
    """
    def __init__(self, emb_dim, return_attn_weights=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.return_attn_weights = return_attn_weights
        
        # Linear projections for Query, Key, Value
        self.query_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.val_proj = nn.Linear(emb_dim, emb_dim, bias=False)

        self.scale = 1.0 / math.sqrt(emb_dim)

    def forward(self, inputs, causal_mask=None, **kwargs):
        # inputs: [B, Seq_len, D]
        Query = self.query_proj(inputs)
        Key = self.key_proj(inputs)
        Val = self.val_proj(inputs)

        # Compute attention scores
        attn_scores = torch.matmul(Query, Key.transpose(-2, -1)) * self.scale
        
        if causal_mask is not None:
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, Val)
        
        if self.return_attn_weights:
            return attn_output, attn_weights
        else:
            return attn_output  

@BlockRegistry.register("head_attention")
class HeadAttn(Block):
    """
    A single attention head often used as a component of Multi-Head Attention.

    This block projects the input to a smaller `head_size` and computes
    self-attention within that subspace.

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        head_size (int): Dimensionality of the head (typically emb_dim // num_heads).
        drop_fact (float): Dropout probability. Defaults to 0.0.
        causal_mask (bool): If True, applies a causal mask for auto-regressive decoding.
        return_attn_weights (bool): If True, returns both the output and the attention weights.
    """
    def __init__(self, emb_dim=256, head_size=16, drop_fact=0.0, causal_mask=False, return_attn_weights=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.head_size = head_size
        self.causal_mask = causal_mask
        self.return_attn_weights = return_attn_weights
        
        self.query_proj = nn.Linear(emb_dim, head_size, bias=False)
        self.key_proj = nn.Linear(emb_dim, head_size, bias=False)
        self.val_proj = nn.Linear(emb_dim, head_size, bias=False)

        self.scale = 1.0 / math.sqrt(head_size)
        self.dropout = nn.Dropout(drop_fact)

    def forward(self, inputs, **kwargs):
        B, seq_len, D = inputs.shape
        Query = self.query_proj(inputs)
        Key = self.key_proj(inputs)
        Val = self.val_proj(inputs)

        scores = torch.matmul(Query, Key.transpose(-2, -1)) * self.scale
        
        if self.causal_mask:
            causal_tril = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=inputs.device))
            scores = scores.masked_fill(causal_tril == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, Val)
        
        if self.return_attn_weights:
            return attn_output, attn_weights
        else:
            return attn_output 

@BlockRegistry.register("multi_head_attn")
class MultiHeadAttn(Block):
    """
    Multi-Head Attention (MHA) block.

    MHA allows the model to jointly attend to information from different
    representation subspaces at different positions. This implementation
    uses a collection of `HeadAttn` blocks.

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        drop_fact (float): Dropout probability. Defaults to 0.0.
        causal_mask (bool): If True, applies causal masking to all heads.
        return_attn_weights (bool): If True, returns both the output and the attention weights.
    """
    def __init__(self, emb_dim=256, num_heads=8, drop_fact=0.0, causal_mask=False, return_attn_weights=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        
        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")

        self.head_size = emb_dim // num_heads
        self.heads = nn.ModuleList([
            HeadAttn(emb_dim, head_size=self.head_size, drop_fact=drop_fact, 
                     causal_mask=causal_mask, return_attn_weights=return_attn_weights) 
            for _ in range(num_heads)
        ])
        
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(drop_fact)

    def forward(self, inputs, **kwargs):
        # inputs: [B, Seq_len, D]
        head_outputs = [head(inputs) for head in self.heads]
        
        # Concatenate head outputs
        concat_output = torch.cat(head_outputs, dim=-1)
        
        # Final linear projection
        output = self.proj(concat_output)
        output = self.dropout(output)
        
        return output
