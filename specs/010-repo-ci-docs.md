# 010 — Repo, tooling, CI, docs

## Requirements

- **FR-REPO-1** Remove `.vscode/`, `src/model_2.py`, `setup.py`, `requirements.txt`,
  `MANIFEST.in`, and all Italian references.
- **FR-REPO-2** `pyproject.toml` is the single packaging/config source:
  - package `pytorch_transformers` under `src/`
  - deps: `torch>=2.4`, `datasets`, `tokenizers`, `safetensors`, `sacrebleu`,
    `torchmetrics`, `tensorboard`, `typer`, `rich`, `pyyaml`, `tqdm`
  - dev deps: `pytest`, `ruff`, `mypy`
  - `[tool.ruff]`, `[tool.pytest.ini_options]`, `[tool.mypy]` configured
- **FR-REPO-3** `Makefile` thin targets: `install`, `lint`, `format`, `test`, `specs`,
  `train-tokenizer`, `prepare-data`, `train`, `eval`, `translate`, `export`, `bench`,
  `rust-build`, `rust-test`, `rust-build-cuda`.
- **FR-REPO-4** GitHub Actions `.github/workflows/ci.yml`:
  - `python`: ruff + mypy + pytest on tiny synthetic data (CPU only, no dataset downloads)
  - `rust`: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test` (CPU)
  - caches: pip/uv, cargo
- **FR-REPO-5** README: dual-stack intro, architecture diagram references, EN↔JP quickstart
  for Python and Rust, data attribution (JESC CC BY-SA 4.0; OPUS), CLI reference, and the
  benchmark table.
- **FR-REPO-6** `.gitignore` covers `data/`, `weights/`, `runs/`, `bench/results.json`,
  `target/`, tokenizer JSONs.
- **FR-REPO-7** `CONTRIBUTING.md` template placeholders cleaned; no dangling
  `CODE_OF_CONDUCT.md` link unless the file exists.

## Acceptance

- `make test` and `make lint` pass locally.
- CI passes on a clean clone without GPU or dataset access.
- README quickstart commands are copy-paste runnable.