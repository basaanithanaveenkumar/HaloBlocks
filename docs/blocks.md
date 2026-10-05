# Block reference

All 37 registered blocks, generated from `BlockRegistry` and the constructors'
signatures (HaloBlocks v0.1.3). Every block can be created as `blocks.<Key>(...)`,
`hb.create("<Key>", ...)` or `hb.create({"type": "<Key>", ...})`.

| Registry key | Module | Constructor | Summary |
|---|---|---|---|
| `AlibiPositionalBias` | `blocks.positional_embedding.alibi` | `(num_heads, max_len=2048, slope_factor=None)` | Attention with Linear Biases (ALiBi). |
| `CompositeBlock` | `core.composite` | `(blocks: List[haloblocks.core.block.Block])` | A container block that executes a sequence of sub-blocks. |
| `CrossAttention` | `blocks.attention.cross_attention` | `(emb_dim, return_attn_weights=False, use_q_norm=False, use_k_norm=False)` | Single-head cross-attention: keys are projected from ``context``; values use |
| `DecoderTransformer` | `blocks.transformer.decoder` | `(num_layers=16, emb_dim=1024, num_heads=32, mlp_dim=512, drop_fact=0.0, use_moe=True, moe_hid_scale=1.2, moe_num_routed_experts=16, moe_top_k=4, moe_num_shared_experts=2)` | Base class for all composable neural network components in HaloBlocks. |
| `DeepseekMoE` | `blocks.moe.deepseek_moe` | `(emb_dim, hid_dim, num_router_exprts, best_k, num_shared_exprts)` | DeepSeek Mixture-of-Experts (MoE) implementation. |
| `DilatedSlidingWindowAttention` | `blocks.attention.sliding_window_attention` | `(emb_dim=256, num_heads=4, window_size=64, dilation=1, use_q_norm=True, use_k_norm=True, causal=False)` | Dilated Sliding Window Attention with dilation to increase receptive field. |
| `DynamicSlidingWindowAttention` | `blocks.attention.sliding_window_attention` | `(emb_dim=256, num_heads=4, window_size=128, learn_window=True, use_q_norm=True, use_k_norm=True, causal=False)` | Dynamic Sliding Window Attention with learnable window size per head. |
| `ELUFeatureMap` | `blocks.attention.linear_attention` | `(eps=1e-06)` | ELU-based feature map for linear attention. |
| `ExpFeatureMap` | `blocks.attention.linear_attention` | `(*args: Any, **kwargs: Any) -> None` | Exponential feature map for linear attention. |
| `FlowActionDecoder` | `blocks.vla.flow_decoder` | `(action_dim_flat, obs_dim, hidden_dim=1024, time_embed_dim=128)` | Predicts the vector field v_theta(x_t, t, condition). |
| `GatedAttention` | `blocks.attention.gated_attention` | `(emb_dim=256, num_heads=4, use_q_norm=True, use_k_norm=True, gate_bias=0.0)` | Gated Attention mechanism with explicit gating over attention output. |
| `GatedAttentionWithMask` | `blocks.attention.gated_attention` | `(emb_dim=256, num_heads=4, use_q_norm=True, use_k_norm=True, gate_bias=0.0)` | Gated Attention with masking support. |
| `GatedCrossAttention` | `blocks.attention.gated_attention` | `(emb_dim=256, num_heads=4, use_q_norm=True, use_k_norm=True, gate_bias=0.0)` | Gated Cross-Attention mechanism with explicit gating. |
| `GroupedQueryAttention` | `blocks.attention.grouped_query_attention` | `(emb_dim=256, num_heads=8, num_kv_heads=2, dropout=0.0, use_q_norm=False, use_k_norm=False)` | Grouped-Query Attention (GQA) module. |
| `HeadAttention` | `blocks.attention.self_attention` | `(emb_dim=256, head_size=16, drop_fact=0.0, causal_mask=False, return_attn_weights=False, use_q_norm=False, use_k_norm=False)` | A single attention head often used as a component of Multi-Head Attention. |
| `HeadCrossAttention` | `blocks.attention.cross_attention` | `(emb_dim=256, head_size=16, drop_fact=0.0, causal_mask=False, return_attn_weights=False, use_q_norm=False, use_k_norm=False)` | A single cross-attention head. |
| `LearnedPositionalEmbedding` | `blocks.positional_embedding.learned` | `(emb_dim, max_len, dropout=0.0)` | Learned absolute positional embedding. |
| `LinearAttention` | `blocks.attention.linear_attention` | `(emb_dim=256, num_heads=4, feature_map='elu', use_q_norm=True, use_k_norm=True, causal=False, eps=1e-06)` | Linear Attention mechanism with O(n) complexity. |
| `LinearAttentionWithRoPE` | `blocks.attention.linear_attention` | `(emb_dim=256, num_heads=4, feature_map='elu', use_q_norm=True, use_k_norm=True, causal=False, eps=1e-06, rope_base=10000.0)` | Linear Attention with Rotary Position Embeddings. |
| `LinearCrossAttention` | `blocks.attention.linear_attention` | `(emb_dim=256, num_heads=4, feature_map='elu', use_q_norm=True, use_k_norm=True, causal=False, eps=1e-06)` | Linear Cross-Attention with O(n) complexity. |
| `MLP` | `blocks.mlp.mlp` | `(input_dim: int, hidden_dims: List[int], output_dim: Optional[int] = None, activation: str = 'relu', bias: bool = True, last_layer_activation: bool = False, dropout: float = 0.0)` | A configurable Multi-Layer Perceptron (MLP) block. |
| `MultiHeadAttention` | `blocks.attention.self_attention` | `(emb_dim=256, num_heads=8, drop_fact=0.0, causal_mask=False, return_attn_weights=False, use_q_norm=False, use_k_norm=False)` | Multi-Head Attention (MHA) block. |
| `MultiHeadCrossAttention` | `blocks.attention.cross_attention` | `(emb_dim=256, num_heads=8, drop_fact=0.0, causal_mask=False, return_attn_weights=False, use_q_norm=False, use_k_norm=False)` | Multi-Head Cross Attention (MHCA) block. |
| `MultiHeadLatentAttention` | `blocks.attention.multi_head_latent_attention` | `(emb_dim=256, num_heads=8, latent_dim=None, dropout=0.0, use_q_norm=False, use_k_norm=False, tie_kv_down=False)` | Multi-Head Latent Attention with absorption trick. |
| `MultiQueryAttention` | `blocks.attention.multi_query_attention` | `(emb_dim=256, num_heads=8, dropout=0.0, use_q_norm=False, use_k_norm=False)` | Multi-Query Attention. |
| `ReLUFeatureMap` | `blocks.attention.linear_attention` | `(*args: Any, **kwargs: Any) -> None` | ReLU-based feature map for linear attention. |
| `RotaryPositionalEmbedding` | `blocks.positional_embedding.rotary` | `(head_dim, max_len=2048, base=10000.0)` | Rotary Positional Embedding (RoPE). |
| `ScaledDotProductAttention` | `blocks.attention.scaled_dot_product_attention` | `(dropout=0.1, head_dim=None, use_q_norm=False, use_k_norm=False)` | Standard Scaled Dot-Product Attention mechanism. |
| `SelfAttention` | `blocks.attention.self_attention` | `(emb_dim, return_attn_weights=False, use_q_norm=False, use_k_norm=False)` | A basic Single-Head Self-Attention block. |
| `SinusoidalPositionalEmbedding` | `blocks.positional_embedding.sinusoidal` | `(emb_dim, max_len=5000, dropout=0.0)` | Sinusoidal positional embedding (absolute, fixed). |
| `SlidingWindowAttention` | `blocks.attention.sliding_window_attention` | `(emb_dim=256, num_heads=4, window_size=128, use_q_norm=True, use_k_norm=True, causal=False)` | Sliding Window Attention mechanism with local attention patterns. |
| `StackedTransformerBlock` | `core.builder` | `(emb_dim: int = 256, num_layers: int = 1, attn: Union[NoneType, str, dict, haloblocks.core.block.Block] = None, attn_kwargs: Optional[Dict[str, Any]] = None, cross_attn: Union[NoneType, str, dict, haloblocks.core.block.Block] = None, cross_attn_kwargs: Optional[Dict[str, Any]] = None, ffn: Union[NoneType, str, dict, haloblocks.core.block.Block] = None, ffn_kwargs: Optional[Dict[str, Any]] = None, norm: str = 'layernorm', drop_fact: float = 0.0, operation_order: Optional[tuple] = None)` | Multi-layer transformer stack, constructible from a config dict. |
| `StackedTransformerBlocks` | `core.builder` | `(layers: torch.nn.modules.container.ModuleList, emb_dim: int, norm_template: torch.nn.modules.module.Module)` | A stack of ``TransformerBlockBuilder`` layers with a final norm. |
| `TransformerBlock` | `blocks.transformer.transformer_block` | `(emb_dim=256, num_heads=8, mlp_dim=512, drop_fact=0.0, causal_mask=False, use_moe=True, moe_hid_scale=1.2, moe_num_routed_experts=16, moe_top_k=4, moe_num_shared_experts=2)` | Base class for all composable neural network components in HaloBlocks. |
| `TransformerBlockBuilder` | `core.builder` | `(emb_dim: int = 256, attn: Union[NoneType, str, dict, haloblocks.core.block.Block] = None, attn_kwargs: Optional[Dict[str, Any]] = None, cross_attn: Union[NoneType, str, dict, haloblocks.core.block.Block] = None, cross_attn_kwargs: Optional[Dict[str, Any]] = None, ffn: Union[NoneType, str, dict, haloblocks.core.block.Block] = None, ffn_kwargs: Optional[Dict[str, Any]] = None, norm: str = 'layernorm', drop_fact: float = 0.0, operation_order: Optional[tuple] = None)` | A composable transformer layer built from user-selected components. |
| `TrinityAttention` | `blocks.attention.trinity_attention` | `(emb_dim=256, num_heads=4, use_q_norm=True, use_k_norm=True)` | Trinity Attention mechanism. |
| `TrinityCrossAttention` | `blocks.attention.trinity_attention` | `(emb_dim=256, num_heads=4, use_q_norm=True, use_k_norm=True)` | Trinity Cross-Attention mechanism. |

## Notes

- `TrinityAttention` / `TrinityCrossAttention`: softmax attention with per-head RMSNorm on
  queries and keys (on by default) and a **sigmoid output gate** `σ(W_g x) ⊙ attn`. It takes
  no mask argument. (The README's "local + global + linear" description does not match
  the code.)
- `MultiHeadLatentAttention`: shared low-rank KV down-projection (`latent_dim`, default
  `emb_dim // 4`) with per-head up-projections, using the absorption trick for scores.
- `RotaryPositionalEmbedding` takes `head_dim`, not `emb_dim`.
- `TransformerBlockBuilder(ffn=...)`: use the **dict** form for `DeepseekMoE`
  (`{"type": "DeepseekMoE", "emb_dim": ..., ...}`). The string form plus `ffn_kwargs`
  currently raises `unexpected keyword argument 'input_dim'`.
