# 004 — Training

## Requirements

- **FR-TRAIN-1** Configs: `configs/en-jp.yaml`, `configs/jp-en.yaml`, `configs/dev.yaml`
  (tiny subset, few steps). YAML is the single source of truth; CLI flags override.
- **FR-TRAIN-2** Optimization, paper-style:
  - Adam `β=(0.9, 0.98)`, `eps=1e-9`
  - LR schedule: linear warmup (`warmup_steps=4000`) then inverse-sqrt decay
  - label smoothing `0.1`; `ignore_index=[PAD]`
  - gradient clipping (global norm `1.0`)
- **FR-TRAIN-3** Mixed precision: bf16 autocast on CUDA (GradScaler if fp16 requested);
  no-op on CPU/MPS.
- **FR-TRAIN-4** Device resolution: `auto` → `cuda` if available → `mps` → `cpu`;
  `--device` overrides. `cuda` device properties are logged (name, memory).
- **FR-TRAIN-5** Checkpointing: safetensors weights + optimizer state + `metadata.json`
  `{epoch, global_step, direction, tokenizer_sha256, git_rev, config}`. Save `last`
  every `checkpoint_interval` steps and every epoch, `best` by validation loss, and
  retained step snapshots every `snapshot_interval` (pruned to `keep_checkpoints`).
  Resume must call `model.load_state_dict` and restore `global_step` (fixes the original bug).
- **FR-TRAIN-6** Logging: stdlib `logging` + TensorBoard scalars
  (`train/loss`, `train/lr`, `train/tokens_per_s`, `val/loss`, `val/chrf`, `val/bleu`);
  optional `--wandb` (off by default).
- **FR-TRAIN-7** Validation every `val_interval` steps: loss + chrF/BLEU on a capped
  sample + a few printed translations.
- **FR-TRAIN-8** Reproducibility: seed Python/NumPy/Torch; deterministic DataLoader worker
  seeding. Optional `--deterministic` for strict algorithms.
- **FR-TRAIN-9** Data loading: `num_workers`, `pin_memory`, `persistent_workers`,
  prefetch; tokens/sec counter from the actual `(B, S+T)` consumed.
- **FR-TRAIN-10** `direction: mixed` trains a single model over both orientations using language embeddings (`lang_embedding: true`). At eval time, select a direction with `--direction en-jp|jp-en`.
- **FR-TRAIN-11** Memory guards: `eval_batch_size` decouples validation memory from the
  training `batch_size`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set unless
  the caller overrides it; a CUDA OOM during a train step retries that step with a halved
  micro-batch (down to 1) instead of aborting; a validation OOM is skipped; and any
  interruption saves a checkpoint before propagating.

## Acceptance

- `train --config configs/dev.yaml` runs end-to-end on a subset, loss decreases, artifacts
  appear, no CUDA errors.
- Interrupt + `--resume last` continues from the correct `global_step`.
- Best checkpoint is selected by validation loss.