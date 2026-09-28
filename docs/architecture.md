# Architecture

Mermaid diagrams (they render on GitHub). The same figures are used on the
[project page](../project-page/index.html) and in the [paper](../paper/main.tex).

## 1. One registry, three ways in

```mermaid
flowchart TB
  A1["blocks.GroupedQueryAttention(...)"] --> REG["BlockRegistry<br/>class name → class"]
  A2["hb.create('GroupedQueryAttention', ...)"] --> REG
  A3["hb.create({'type': 'GroupedQueryAttention', ...})"] --> FAC["BlockFactory.create<br/>recursive for 'blocks'"]
  FAC --> REG
  REG --> ATT["attention (24)"]
  REG --> POS["positions (4)"]
  REG --> FF["MLP · DeepseekMoE"]
  REG --> TR["composition (6)"]
  REG --> VLA["FlowActionDecoder"]
```

## 2. TransformerBlockBuilder

```mermaid
flowchart LR
  SPEC["attn / cross_attn / ffn spec<br/>None · 'Name' · {'type': …} · Block"] --> RES["resolve → Block<br/>(inject emb_dim if accepted)"]
  RES --> ORD{"operation_order"}
  ORD -->|PRE_NORM| P["(norm, self_attn) → (norm, ffn)"]
  ORD -->|POST_NORM| Q["(self_attn, norm) → (ffn, norm)"]
  ORD -->|PRE_NORM_CROSS| R["(norm, self_attn) → (norm, cross_attn) → (norm, ffn)"]
  P & Q & R --> OUT["residual sublayers<br/>norm: layernorm or rmsnorm"]
  OUT --> STK["StackedTransformerBlock × num_layers"]
```

## 3. Multi-head latent attention (absorption)

```mermaid
flowchart LR
  X["x"] --> DKV["W_DKV: down-project<br/>to latent c (d/4)"]
  X --> WQ["W_Q → q per head"]
  WQ --> ABS["absorb: q̃ = W_UKᵀ q"]
  DKV --> S["scores = q̃ · c"]
  ABS --> S
  S --> SM["softmax"]
  DKV --> UV["W_UV: values per head"]
  SM --> O["Σ weights · values → W_O"]
  UV --> O
```

## 4. Gated and Trinity attention

```mermaid
flowchart LR
  X["x"] --> QKV["W_Q, W_K, W_V"]
  QKV --> N["RMSNorm on q, k per head<br/>(Trinity default)"]
  N --> A["softmax(q kᵀ / √d) v"]
  X --> G["σ(W_g x + b_g)"]
  A --> M["⊙"]
  G --> M
  M --> O["W_O"]
```

## 5. DeepSeek MoE

```mermaid
flowchart TB
  X["x [B, T, D]"] --> SH["shared SwiGLU experts (always on)"]
  X --> R["noisy top-k router<br/>logits + softplus-scaled noise (train)"]
  R --> E["k routed SwiGLU experts, gate-weighted"]
  X --> E
  SH --> S["Σ"]
  E --> S
```

## 6. Flow-matching action decoder (VLA)

```mermaid
flowchart LR
  XT["noisy action chunk x_t"] --> C["concat"]
  T["t ∈ [0,1] → sinusoidal embedding"] --> C
  OBS["observation embedding"] --> C
  C --> MLP["MLP (ReLU, hidden 1024)"] --> V["velocity v(x_t, t, obs)"]
  V --> ODE["inference: Euler steps from noise (t = 0) to action (t = 1)"]
```
