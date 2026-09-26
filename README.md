# pytorch-transformers

A from-scratch **Transformer** (Vaswani et al., 2017 — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)) for **English ↔ Japanese** translation, implemented twice:

- **Python / PyTorch** — the reference implementation, used for training and evaluation.
- **Rust / [candle](https://github.com/huggingface/candle)** — an independent port for fast inference and training.

Both stacks share one tokenizer (`tokenizer.json`) and one weight format (`safetensors`), so their outputs and speed are directly comparable.

> Educational project. Nothing here is intended for production use.

## Features

- Paper-faithful encoder–decoder with multi-head attention, sinusoidal positional encodings, and configurable post-/pre-norm.
- Shared byte-level BPE tokenizer (works for Japanese without whitespace assumptions).
- Two-source data pipeline: **JESC** (conversational subtitles) and **OPUS-100** (mixed domain), with deterministic mixing, filtering, dedup, decontamination, and dual-track evaluation.
- Directional configs for **EN→JA**, **JA→EN**, and a **mixed** direction model using shared language embeddings.
- Training with warmup + inverse-sqrt LR, label smoothing, AMP, gradient clipping, resume, periodic/pruned step snapshots, best/last checkpoints, and OOM guards (halved micro-batch retry, smaller eval batch, `expandable_segments`).
- Greedy and beam-search decoding; BLEU and chrF evaluation.
- `safetensors` export with a frozen key contract for Python ↔ Rust interop.
- A Python/Rust parity and benchmark harness.

## Repository layout

```
specs/                        design specs for every component
configs/                      data + training configs (YAML)
src/pytorch_transformers/     Python package
rust/                         Rust/candle crate (binary: ptr)
scripts/                      tokenizer/data, parity fixture, comparison harness
tests/                        Python tests
docs/                         the paper and architecture figures
```

## Setup (Python)

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

If you have a CUDA GPU, the default PyPI wheels are CUDA-enabled. Device selection is automatic: `cuda → mps → cpu`; override with `--device`.

## Python quickstart

```bash
# 1. train a shared tokenizer on a sample of both corpora
ptx train-tokenizer configs/data.yaml --out tokenizer_shared.json

# 2. prepare data (streams OPUS-100 + JESC, mixes 30/70, exports shards)
ptx prepare-data configs/data.yaml --tokenizer tokenizer_shared.json --out data/opus_jesc

# 3. train a direction
ptx train configs/en-jp.yaml --device auto

# 4. evaluate / translate
ptx eval weights/en-jp --split jesc-own-test --beam 4
ptx translate weights/en-jp --text "Hello, how are you?"

# 5. export flat safetensors for the Rust port
ptx export weights/en-jp --out model.safetensors
```

For a fast smoke run, use `configs/dev.yaml` and `--max-examples 2000` on `prepare-data`.

## Rust quickstart

```bash
cd rust
cargo build --release            # CPU
cargo build --release --features cuda   # NVIDIA GPU

# translate with an exported model + config + tokenizer
./target/release/pytorch-transformers-rs translate \
  --model model.safetensors --config model.json \
  --tokenizer tokenizer_shared.json --text "Hello, how are you?"

# benchmark in-process (JSON timings)
./target/release/pytorch-transformers-rs bench \
  --model model.safetensors --config model.json \
  --tokenizer tokenizer_shared.json --text "Hello"

# train from the Python-exported shards
./target/release/pytorch-transformers-rs train \
  --config model.json --data-dir data/opus_jesc \
  --tokenizer tokenizer_shared.json --steps 100 --out rust_run
```

## Parity & benchmark

```bash
make bench     # runs scripts/compare.py: Python vs Rust, checks output parity
```

Parity is enforced by tests: Rust logits match Python within `1e-3`, and greedy decoding produces identical token ids. `bench/results.json` records median/p95 timings for both stacks on the same inputs.

> Note: the committed parity fixture is a tiny random model, so timings are dominated by per-op overhead rather than compute. Use a real exported checkpoint and a larger model for meaningful numbers.

## Data & attribution

- **JESC** — Japanese–English Subtitle Corpus (Pryzant et al., 2018), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- **OPUS-100** — Zhang et al., 2020; released for research use.
- **JLPT study sentences** (optional, local) — grammar examples and Core 2000 sentences from a local `nihongo-go` corpus, mixed at low weight and held out in part as a level-stratified domain eval (`jlpt-grammar-eval`, `jlpt-core2000-eval`). Sources marked `optional: true` are skipped when their paths are absent.

See `specs/001-data.md` for the exact mixing and split policy. Evaluate the JLPT domain with e.g. `ptx eval weights/en-jp --split jlpt-grammar-eval --direction en-jp`.

## Development

```bash
make test      # pytest (tiny synthetic data, no downloads)
make lint      # ruff + mypy
cd rust && cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test
```

## Status / roadmap

- Done: model, tokenizer, streaming two-source data, training, evaluation, CLI, Rust inference (greedy + beam) and training (dropout, warmup + inverse-sqrt LR), mixed-direction language embeddings, parity + benchmark.
- Supported directions: `en-jp`, `jp-en`, and `mixed` (one model, language embeddings). Train mixed with `configs/mixed.yaml`, then evaluate a direction with `ptx eval ... --direction en-jp|jp-en`.
- Planned: full-corpus throughput tuning, Rust `--dtype bf16` option, fp16 `GradScaler` on Python.

## License

MIT. See [LICENSE](LICENSE).
