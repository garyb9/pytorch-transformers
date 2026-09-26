# 001 — Data, two-source mixing, and splits

## Sources

| id | hub_id | config | fields | native splits | rows | license | domain |
|---|---|---|---|---|---|---|---|
| `opus100` | `Helsinki-NLP/opus-100` | `en-ja` | `translation.en`, `translation.ja` | train/validation/test | 1M / 2k / 2k | unknown (research) | mixed web+books |
| `jesc` | `nntsuzu/JESC` | default | `translation.en`, `translation.ja` | train only (raw corpus) | 2,801,388 | CC BY-SA 4.0 | conversational subtitles |
| `jesc-official` | `nlp.stanford.edu/projects/jesc/data/split.tar.gz` | — | parallel TSV | train 2,797,388 / dev 2000 / test 2000 | — | CC BY-SA 4.0 | same, official splits |

The HF `jesc` mirror is the raw 2019-de-duplicated corpus (matches the official "Raw"
count). Official dev/test are used as an **external benchmark only**.

## Requirements

- **FR-DATA-1** Sources are declared in YAML (`configs/data.yaml`): `{id, hub_id, config,
  split_map, weight, max_examples, license, attribution}`. Adding a source requires no code change.
- **FR-DATA-2** Deterministic mixing: explicit weights (`jesc: 0.7`, `opus100: 0.3`),
  `temperature: 1.0` default. `max_examples` caps per-source rows for dev runs.
  Mixing is a seeded sample over the union of normalized, filtered pairs.
- **FR-DATA-3** Normalization: Unicode NFKC, strip, collapse runs of whitespace.
  JP punctuation and scripts are preserved; no language-specific stripping.
- **FR-DATA-4** Filters: non-empty both sides; token length in `[1, seq_len-2]`;
  target/source token-length ratio ≤ 3.
- **FR-DATA-5** Dedup: exact hash of normalized `(src, tgt)` across all sources.
  (OPUS includes subtitle data, so JESC overlap is expected.)
- **FR-DATA-6** Decontamination: drop any train pair whose normalized source or target
  appears in any eval split: OPUS test/validation, JESC-own test/dev, official JESC dev/test.
- **FR-DATA-7** Direction augmentation: each pair yields both orientations once;
  `direction` is selected at train time. One prepared corpus serves `en-jp` and `jp-en`.
- **FR-DATA-8** Export: sharded JSONL `{"src":[ids],"tgt":[ids],"origin":"...","pair_id":N}`
  plus `manifest.json` with `{counts, max_lens, weights, tokenizer_sha256, git_rev, seed,
  stats}`. Rust consumes exactly these files.
- **FR-DATA-9** `data/stats.json`: per-source and combined length histograms, token totals,
  filter drop counts, dedup/decontamination counts.
- **FR-DATA-10** Dual-track JESC evaluation:
  - **Track A (primary):** carve a seeded 2k dev / 2k test from the raw HF corpus as part
    of our pipeline.
  - **Track B (external):** if `data/raw/jesc-official/` (or `--jesc-official <path>`)
    is present, use the official dev/test for reporting; never train on them.
- **FR-DATA-11** README records attribution: JESC (CC BY-SA 4.0, Pryzant et al. 2018),
  OPUS-100 (Zhang et al. 2020; research use).

## Non-functional

- **NFR-DATA-1** `prepare-data` streams: only eval sets, dedup hashes, and the output buffer
  are held in memory; training examples are emitted as they are produced.
- **NFR-DATA-2** Byte-identical reruns for a fixed seed and source revisions.

## Acceptance

- Rerunning `prepare-data` yields identical manifests and hashes.
- A training batch can be tagged by origin; both origins appear at roughly the target ratio.
- Per-source and combined metrics are reported by `eval`.