# 009 — Parity & benchmark

## Parity

- **FR-BENCH-1** `scripts/compare.py` orchestrates both stacks on identical inputs and
  checkpoint.
- **FR-BENCH-2** Forward parity: fixed input batch + exported weights, compare logits
  `max|Δ| ≤ 1e-3` (f32, CPU) or relative tolerance on GPU; report the actual max delta.
- **FR-BENCH-3** Decode parity: greedy hypotheses must match token-for-token on a fixed
  sample; beam results compared structurally (score within tolerance).
- **FR-BENCH-4** Tokenizer parity: identical ids for a fixed sentence set in both languages.

## Benchmark

- **FR-BENCH-5** Methodology: ≥3 warmups, ≥5 measured reps, report median and p95.
- **FR-BENCH-6** Metrics: cold start (process → first token), time-to-first-token, decode
  tokens/s (greedy, beam 4), batch throughput (sentences/s at fixed batch), peak GPU memory
  (`torch.cuda.max_memory_allocated` / `nvidia-smi` for Rust), CPU wall time.
- **FR-BENCH-7** Devices: run on `cuda` and `cpu` for both stacks; record GPU model/driver.
- **FR-BENCH-8** Output: `bench/results.json` + a markdown table inserted into the README
  (or `bench/RESULTS.md`), including hardware and versions.
- **FR-BENCH-9** `make bench` runs it end-to-end.

## Acceptance

- Parity checks pass before any timing numbers are reported.
- Benchmark table reproducible within reasonable variance on the same machine.