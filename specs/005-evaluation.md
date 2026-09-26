# 005 — Evaluation

## Requirements

- **FR-EVAL-1** Metrics via `sacrebleu`: BLEU (spBLEU optional) and chrF. chrF is primary
  for JA because whitespace-based WER/CER is misleading for Japanese.
- **FR-EVAL-2** Decoding: greedy and beam search (`beam_size` 1/4/5, `length_penalty`).
- **FR-EVAL-3** Reporting granularity: per source (`opus100` test, `jesc` own test,
  `jesc-official` test when available) and combined; plus validation perplexity.
- **FR-EVAL-4** `eval` writes `results.json` and prints a markdown table.
- **FR-EVAL-5** `--direction` selects the model direction; a direction mismatch between
  checkpoint metadata and request is a hard error.
- **FR-EVAL-6** `--limit N` for quick runs; `--beam` override; `--device`.

## Acceptance

- Same checkpoint + split + decode settings produce identical metrics across runs (seeded).
- Results include BLEU, chrF, sentence counts, and decode settings for reproducibility.