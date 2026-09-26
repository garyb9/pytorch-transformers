# 002 — Shared tokenizer

## Requirements

- **FR-TOK-1** Byte-level BPE via the HuggingFace `tokenizers` library (Python) and
  `tokenizers` crate (Rust) — the *same* implementation family in both languages.
- **FR-TOK-2** Vocab size 32k (configurable). Special tokens, fixed ids:
  `[PAD]=0`, `[UNK]=1`, `[BOS]=2`, `[EOS]=3`. Reserved for future mixed-direction mode:
  `<2en>=4`, `<2jp>=5` (trained in, unused by default).
- **FR-TOK-3** Trained on a seeded sample of the union of sources (both languages) so the
  pair shares one vocabulary.
- **FR-TOK-4** Persisted as `tokenizer.json`; its SHA-256 is recorded in every manifest,
  checkpoint metadata, and exported `config.json`.
- **FR-TOK-5** CLI: `train-tokenizer`; `tokenize` smoke command for debugging.
- **FR-TOK-6** Parity: Python and Rust load the same file and produce identical ids for a
  fixed sentence set (tested in both test suites).
- **FR-TOK-7** Encoding adds `[BOS]`/`[EOS]` at the dataset boundary, not inside the
  tokenizer, so truncation/padding stays in the dataset layer.

## Acceptance

- `decode(encode(x))` preserves normalized text (modulo byte-level whitespace artifacts).
- Zero `[UNK]` tokens on a sample of JP sentences.
- Tokenizer hash mismatch between checkpoint and config is a hard error.