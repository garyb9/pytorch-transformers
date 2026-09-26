from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .checkpoints import load_training_checkpoint, save_training_checkpoint
from .config import ModelConfig
from .dataset import TranslationDataset
from .metrics import corpus_chrf
from .model import Transformer, build_transformer, count_parameters
from .tokenizer import TokenizerWrapper
from .translate import content_ids, lang_ids_for, translate_ids
from .utils import build_optimizer, build_scheduler, resolve_device, set_seed

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainConfig:
    direction: str = "en-jp"
    data_dir: str = "data/opus_jesc"
    tokenizer_path: str = "tokenizer_shared.json"
    model_folder: str = "weights"
    run_name: str = "en-jp"
    seq_len: int = 256
    batch_size: int = 32
    num_epochs: int = 20
    lr: float = 5e-4
    warmup_steps: int = 4000
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    residual_mode: str = "post"
    tie_embeddings: bool = False
    lang_embedding: bool = False
    amp: bool = True
    num_workers: int = 4
    eval_batch_size: int = 16
    checkpoint_interval: int = 1000
    snapshot_interval: int = 5000
    keep_checkpoints: int = 3
    val_interval: int = 1000
    val_batches: int = 20
    seed: int = 42
    resume: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in raw.items() if key in known})

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def model_config(self, src_vocab_size: int, tgt_vocab_size: int) -> ModelConfig:
        return ModelConfig(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            src_seq_len=self.seq_len,
            tgt_seq_len=self.seq_len,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            residual_mode=self.residual_mode,
            tie_embeddings=self.tie_embeddings,
            lang_embedding=self.lang_embedding or self.direction == "mixed",
        )


def load_manifest(data_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


_OOM_ERRORS: tuple[type[BaseException], ...] = (
    (torch.cuda.OutOfMemoryError,) if hasattr(torch.cuda, "OutOfMemoryError") else ()
)


def _forward_loss(
    model: Transformer,
    batch: dict[str, torch.Tensor],
    loss_fn: nn.Module,
    use_amp: bool,
    device: torch.device,
) -> torch.Tensor:
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        encoder_output = model.encode(
            batch["encoder_input"], batch["encoder_mask"], batch["src_lang_id"]
        )
        decoder_output = model.decode(
            encoder_output,
            batch["encoder_mask"],
            batch["decoder_input"],
            batch["decoder_mask"],
            batch["tgt_lang_id"],
        )
        logits = model.project(decoder_output)
        return loss_fn(logits.reshape(-1, logits.size(-1)), batch["label"].reshape(-1))


def _empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _train_step(
    model: Transformer,
    moved: dict[str, torch.Tensor],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainConfig,
    use_amp: bool,
    device: torch.device,
) -> tuple[float, int]:
    active = moved
    while True:
        try:
            loss = _forward_loss(model, active, loss_fn, use_amp, device)
            loss.backward()
            if config.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            return float(loss.item()), int(active["encoder_input"].size(0))
        except _OOM_ERRORS:
            optimizer.zero_grad(set_to_none=True)
            _empty_cache(device)
            size = int(active["encoder_input"].size(0))
            if size <= 1:
                raise
            half = max(1, size // 2)
            logger.warning("CUDA OOM; retrying step with micro-batch %d instead of %d", half, size)
            active = {key: value[:half] for key, value in active.items()}


def build_loader(
    shards: list[Path],
    tokenizer: TokenizerWrapper,
    config: TrainConfig,
    shuffle: bool,
    batch_size: int | None = None,
) -> DataLoader:
    dataset = TranslationDataset(
        shards, tokenizer, config.seq_len, direction=config.direction, seed=config.seed
    )
    return DataLoader(
        dataset,
        batch_size=batch_size or config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=shuffle,
    )


@torch.no_grad()
def run_validation(
    model: Transformer,
    loader: DataLoader,
    tokenizer: TokenizerWrapper,
    loss_fn: nn.Module,
    device: torch.device,
    config: TrainConfig,
    use_amp: bool,
    val_samples: int = 4,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    batches = 0
    hypotheses: list[str] = []
    references: list[str] = []

    for batch in loader:
        moved = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            encoder_output = model.encode(
                moved["encoder_input"], moved["encoder_mask"], moved["src_lang_id"]
            )
            decoder_output = model.decode(
                encoder_output,
                moved["encoder_mask"],
                moved["decoder_input"],
                moved["decoder_mask"],
                moved["tgt_lang_id"],
            )
            logits = model.project(decoder_output)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), moved["label"].reshape(-1))
        total_loss += float(loss.item())
        batches += 1

        if len(hypotheses) < val_samples:
            source_ids = content_ids(moved["encoder_input"][0].tolist(), tokenizer)
            src_lang, tgt_lang = lang_ids_for(config.direction)
            predicted = translate_ids(
                model,
                tokenizer,
                source_ids,
                device,
                config.seq_len,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )
            hypotheses.append(tokenizer.decode(predicted))
            references.append(tokenizer.decode(content_ids(moved["label"][0].tolist(), tokenizer)))

        if batches >= config.val_batches:
            break

    metrics = {"loss": total_loss / max(1, batches)}
    if hypotheses:
        metrics["chrf"] = corpus_chrf(hypotheses, references)
    model.train()
    return metrics


def train_model(
    config: TrainConfig,
    device_name: str = "auto",
    max_steps: int | None = None,
) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_seed(config.seed)
    device = resolve_device(device_name)
    logger.info("device: %s", device)

    tokenizer = TokenizerWrapper.from_file(config.tokenizer_path)
    manifest = load_manifest(config.data_dir)
    vocab_size = int(manifest["vocab_size"])
    model_config = config.model_config(vocab_size, vocab_size)
    model = build_transformer(model_config).to(device)
    logger.info("parameters: %d", count_parameters(model))

    train_shards = [Path(config.data_dir) / name for name in manifest["shards"]["train"]]
    val_key = next(
        (key for key in ("jesc-own-dev",) if key in manifest["shards"] and manifest["shards"][key]),
        None,
    )
    val_loader = None
    if val_key is not None:
        val_shards = [Path(config.data_dir) / name for name in manifest["shards"][val_key]]
        val_loader = build_loader(
            val_shards, tokenizer, config, shuffle=False, batch_size=config.eval_batch_size
        )

    train_loader = build_loader(train_shards, tokenizer, config, shuffle=True)
    optimizer = build_optimizer(model, config.lr)
    scheduler = build_scheduler(optimizer, config.warmup_steps)
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id, label_smoothing=config.label_smoothing
    )

    run_dir = Path(config.model_folder) / config.run_name
    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    if config.resume:
        resume_dir = Path(config.resume)
        if resume_dir.is_dir():
            metadata = load_training_checkpoint(resume_dir, model, optimizer)
        else:
            metadata = load_training_checkpoint(run_dir, model, optimizer)
        start_epoch = int(metadata["epoch"]) + 1
        global_step = int(metadata["global_step"])
        best_val = float(metadata.get("best_val", best_val))
        logger.info("resumed from %s at step %d", resume_dir, global_step)

    use_amp = config.amp and device.type == "cuda"
    stopping = False
    epoch = start_epoch
    try:
        for epoch in range(start_epoch, config.num_epochs):
            model.train()
            for batch in train_loader:
                moved = move_batch(batch, device)
                loss_value, micro_batch = _train_step(
                    model, moved, loss_fn, optimizer, scheduler, config, use_amp, device
                )
                global_step += 1

                if global_step % 50 == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        "epoch %d step %d loss %.4f lr %.2e batch %d",
                        epoch,
                        global_step,
                        loss_value,
                        lr,
                        micro_batch,
                    )

                if config.checkpoint_interval and global_step % config.checkpoint_interval == 0:
                    _save(
                        config,
                        model,
                        optimizer,
                        model_config,
                        tokenizer,
                        epoch,
                        global_step,
                        best_val,
                    )

                if (
                    config.snapshot_interval
                    and config.keep_checkpoints > 0
                    and global_step % config.snapshot_interval == 0
                ):
                    _save(
                        config,
                        model,
                        optimizer,
                        model_config,
                        tokenizer,
                        epoch,
                        global_step,
                        best_val,
                        subdir=f"step-{global_step:08d}",
                    )
                    _prune_snapshots(config)

                should_validate = (
                    val_loader is not None
                    and config.val_interval
                    and global_step % config.val_interval == 0
                )
                if should_validate and val_loader is not None:
                    _empty_cache(device)
                    try:
                        metrics = run_validation(
                            model, val_loader, tokenizer, loss_fn, device, config, use_amp
                        )
                    except _OOM_ERRORS:
                        _empty_cache(device)
                        logger.warning("validation OOM at step %d; skipping", global_step)
                        metrics = None
                    if metrics is not None:
                        logger.info("validation: %s", metrics)
                        if metrics["loss"] < best_val:
                            best_val = metrics["loss"]
                            _save(
                                config,
                                model,
                                optimizer,
                                model_config,
                                tokenizer,
                                epoch,
                                global_step,
                                best_val,
                                subdir="best",
                            )
                        _empty_cache(device)

                if max_steps is not None and global_step >= max_steps:
                    stopping = True
                    break

            _save(config, model, optimizer, model_config, tokenizer, epoch, global_step, best_val)
            if stopping:
                break
    except BaseException:
        logger.exception("training interrupted; attempting to save a checkpoint")
        try:
            _save(config, model, optimizer, model_config, tokenizer, epoch, global_step, best_val)
        except Exception:
            logger.exception("failed to save checkpoint after interruption")
        raise

    return {"run_dir": str(run_dir), "global_step": global_step, "best_val": best_val}


def _save(
    config: TrainConfig,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    tokenizer: TokenizerWrapper,
    epoch: int,
    global_step: int,
    best_val: float,
    subdir: str = "",
) -> None:
    run_dir = Path(config.model_folder) / config.run_name
    target = run_dir / subdir if subdir else run_dir
    metadata = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
        "direction": config.direction,
        "tie_embeddings": config.tie_embeddings,
        "tokenizer_sha256": tokenizer.sha256(),
        "model_config": model_config.to_dict(),
        "train_config": config.to_dict(),
    }
    save_training_checkpoint(target, model, optimizer, metadata)


def _prune_snapshots(config: TrainConfig) -> None:
    run_dir = Path(config.model_folder) / config.run_name
    snapshots = sorted(path for path in run_dir.glob("step-*") if path.is_dir())
    for old in snapshots[: -config.keep_checkpoints]:
        shutil.rmtree(old, ignore_errors=True)