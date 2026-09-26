# 008 — Rust port (candle)

Crate: `crates/transformer/` (package `pytorch-transformers-rs`), binary `ptr`.

## Requirements

- **FR-RS-1** Modules: `config`, `model`, `attention`, `tokenizer`, `dataset`, `train`,
  `infer`, `bench`, `main` (clap). Model code mirrors Spec 003 exactly.
- **FR-RS-2** Dependencies: `candle-core`, `candle-nn`, `tokenizers`, `safetensors`,
  `serde`, `serde_json`, `clap`, `anyhow`, `memmap2`, `rand`, `indicatif`.
- **FR-RS-3** Device resolution: `auto` uses `Device::cuda_if_available(0)`, else CPU;
  `--device cpu|cuda|metal` overrides. The `cuda` feature is non-default (CUDA-less builds
  work), but the Makefile GPU target and the GPU CI path compile with `--features cuda`.
- **FR-RS-4** Dtypes: f32 default; `--dtype bf16` on CUDA.
- **FR-RS-5** Loads the shared `tokenizer.json`, exported safetensors, and JSONL shards +
  manifest produced by Python. No HF downloads in Rust.
- **FR-RS-6** Training: `VarMap` + `VarBuilder::from_varmap`, `AdamW`, gradient clipping,
  warmup + inverse-sqrt schedule. **Label-smoothed cross-entropy implemented manually**
  (candle's `cross_entropy` has no smoothing) matching PyTorch's formula and `ignore_index`.
- **FR-RS-7** Inference: greedy + beam, identical stopping rules to Python.
- **FR-RS-8** CLI mirrors Spec 006 (translate/eval/train/bench), same flag names where
  meaningful.
- **FR-RS-9** Errors via `anyhow` with context; no panics on malformed inputs.

## Acceptance

- `cargo test` (CPU, tiny synthetic): shapes, mask, forward parity vs Python (logits
  ≤1e-3), greedy-id parity, key coverage.
- `cargo run --features cuda -- translate ...` runs on the RTX 4070.
- Same checkpoint produces the same greedy translations as Python.