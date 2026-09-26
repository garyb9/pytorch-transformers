# 003 — Model (paper-faithful, fixed)

Reference: Vaswani et al., 2017. Modules are hand-written; no HF model classes.

## Fixes over the original playground code

| id | Original defect | Fix |
|---|---|---|
| FX-1 | `Transformer.encode` called itself → infinite recursion | call `self.encoder(...)` |
| FX-2 | `FeedForward.linear_2` was `d_model→d_ff` | `d_ff→d_model` |
| FX-3 | scalar LayerNorm `alpha`/`bias` | per-feature `nn.LayerNorm(d_model, eps=1e-6)` |
| FX-4 | positional buffer touched with `requires_grad_` in forward | non-persistent buffer, no grad flag |
| FX-5 | projection applied `log_softmax` | return **logits**; loss does softmax |
| FX-6 | `Transformer.encode`/`decode`/`project` API inconsistent | single `forward(src, tgt, src_mask, tgt_mask)` plus `encode`/`decode` used by inference |

## Requirements

- **FR-MODEL-1** `InputEmbedding`: id → vector, scaled by `sqrt(d_model)`
  (subclass of `nn.Embedding` so the parameter key is `*.weight`).
- **FR-MODEL-2** `PositionalEncoding`: sinusoidal, registered as non-persistent buffer,
  added then dropout. Supports sequences up to `seq_len`.
- **FR-MODEL-3** `FeedForward(d_model, d_ff, dropout)`: `Linear→ReLU→Dropout→Linear`.
- **FR-MODEL-4** `MultiHeadAttention(d_model, h, dropout)`: `w_q/w_k/w_v/w_o = Linear(d,d)`;
  heads reshape `(B,L,d)→(B,h,L,d_k)`; scores `QKᵀ/√d_k`; mask fill `-1e9`; softmax;
  dropout; concat; project. Exposes latest attention weights for inspection.
- **FR-MODEL-5** `residual_mode ∈ {post, pre}`, default `post` (paper):
  - post: `LayerNorm(x + Dropout(Sublayer(x)))`
  - pre: `x + Dropout(Sublayer(LayerNorm(x)))`
  Encoder/decoder end with a final `LayerNorm` (post mode).
- **FR-MODEL-6** `Projection(d_model, vocab)`: `Linear`, returns logits.
- **FR-MODEL-7** `build_transformer(config)`: assembles N encoder/decoder blocks,
  Xavier-uniform init for `dim>1`, returns model + logs parameter count.
- **FR-MODEL-8** Optional hooks, off by default: `tie_embeddings` (share src/tgt/proj
  weights if vocab matches) and `lang_embedding` (adds learned language vector; reserved
  for `mixed` direction).

## Tensor contract

| tensor | shape |
|---|---|
| `src` / `tgt` ids | `(B, S)` / `(B, T)` int64 |
| `src_mask` | `(B, 1, 1, S)` bool, True = attend |
| `tgt_mask` | `(B, 1, T, T)` bool, causal ∧ pad |
| encoder output | `(B, S, d_model)` |
| decoder output | `(B, T, d_model)` |
| logits | `(B, T, vocab)` |

## Acceptance

- Shape tests for every module and the full forward pass.
- Causal mask: position `i` cannot attend to `j > i` (verified numerically).
- Padding masked with `-1e9` produces zero attention weight at pad positions.
- Tiny-batch overfit: a 2-layer model on a synthetic copy task reaches loss < 0.05,
  proving gradients flow through encoder→decoder→projection.
- Both `post` and `pre` modes instantiate and run forward/backward.