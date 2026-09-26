# 007 — Checkpoints & Python↔Rust interop

## Safetensors key contract (frozen)

Both implementations must produce/consume exactly these names.

```
src_embed.weight                    tgt_embed.weight
encoder.norm.weight                 encoder.norm.bias
encoder.layers.{i}.norm1.weight     encoder.layers.{i}.norm1.bias
encoder.layers.{i}.norm2.weight     encoder.layers.{i}.norm2.bias
encoder.layers.{i}.self_attn.w_q.{weight,bias}
encoder.layers.{i}.self_attn.w_k.{weight,bias}
encoder.layers.{i}.self_attn.w_v.{weight,bias}
encoder.layers.{i}.self_attn.w_o.{weight,bias}
encoder.layers.{i}.ffn.linear1.{weight,bias}
encoder.layers.{i}.ffn.linear2.{weight,bias}
decoder.norm.weight                 decoder.norm.bias
decoder.layers.{i}.norm1.{weight,bias}
decoder.layers.{i}.norm2.{weight,bias}
decoder.layers.{i}.norm3.{weight,bias}
decoder.layers.{i}.self_attn.w_q.{weight,bias}   (same for w_k, w_v, w_o)
decoder.layers.{i}.cross_attn.w_q.{weight,bias}  (same for w_k, w_v, w_o)
decoder.layers.{i}.ffn.linear1.{weight,bias}
decoder.layers.{i}.ffn.linear2.{weight,bias}
tgt_proj.weight                     tgt_proj.bias
```

Positional encodings are non-persistent buffers and are **not** in the state dict.

## Requirements

- **FR-CKPT-1** Python `export` maps a training checkpoint to a flat safetensors file using
  the names above, plus `config.json` `{d_model, N, h, d_ff, dropout, seq_len, vocab_size,
  direction, residual_mode, tokenizer_sha256, tie_embeddings}`.
- **FR-CKPT-2** `tie_embeddings: true` exports a single `src_embed.weight` and Rust reuses
  it; the contract marks shared tensors explicitly.
- **FR-CKPT-3** A key-coverage test asserts the exported tensor set exactly equals the
  contract (no missing, no extras) for both `tie_embeddings` values.
- **FR-CKPT-4** Direction + tokenizer hash mismatches are hard errors at load time.
- **FR-CKPT-5** PyTorch training checkpoints remain self-contained (`weights.safetensors`,
  `optimizer.pt`, `metadata.json`) under `weights/<run>/`.

## Acceptance

- `export` output loads in candle via `VarBuilder::from_mmaped_safetensors` with no missing
  or unused tensors.
- Round-trip test: model → export → Rust load → identical logits (≤1e-3 f32).