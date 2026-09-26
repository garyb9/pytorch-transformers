# 000 — Constitution

Project: `pytorch-transformers` — an educational, from-scratch Transformer
([Attention Is All You Need, 1706.03762](https://arxiv.org/abs/1706.03762))
implemented twice: Python (reference) and Rust/candle (systems).

## Principles

- **C1 — Correctness over cleverness.** Every behavior we claim is backed by a test.
- **C2 — Paper fidelity.** The model code is written from the paper; no HF `transformers`
  model classes in `model` code. External libraries are limited to tensor/linalg,
  tokenization, data loading, metrics.
- **C3 — Dual implementation, shared artifacts.** Python and Rust share one
  `tokenizer.json` and one safetensors weight contract, so outputs and speed are
  directly comparable.
- **C4 — EN↔JP only.** All Italian/`opus_books` artifacts are removed.
- **C5 — GPU-first.** Device resolution is `cuda → mps → cpu`; an explicit override is
  always available (`--device`). Rust's `cuda` feature is non-default so CUDA-less
  machines still build, but the Makefile/CI GPU path builds with it.
- **C6 — Reproducibility.** Seeds, manifest hashes, tokenizer hash, and pinned toolchains.
- **C7 — Spec-first workflow.** `specs/` describe intent; code follows. No commits unless
  explicitly requested.

## Scope

- Training a Transformer encoder–decoder for EN↔JP translation.
- Two direction configs now (`en-ja`, `ja-en`); mixed-direction is plumbed but disabled.
- Evaluation, greedy/beam inference, checkpoint export, Python↔Rust parity and benchmarks.

## Non-goals

- Beating production MT systems.
- Using pretrained transformer weights.
- Full multilingual support beyond the EN↔JP pair (reserved language tags exist for later).