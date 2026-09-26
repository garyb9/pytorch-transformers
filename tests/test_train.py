from __future__ import annotations

import json
from pathlib import Path

import torch

from pytorch_transformers.checkpoints import load_run_dir
from pytorch_transformers.data import TokenizedPair, write_shards
from pytorch_transformers.eval import evaluate
from pytorch_transformers.tokenizer import TokenizerWrapper, train_tokenizer
from pytorch_transformers.train import TrainConfig, train_model
from pytorch_transformers.utils import build_optimizer, build_scheduler

CORPUS = ["hello world", "こんにちは世界", "the quick brown fox", "日本語のテスト"] * 5


def test_scheduler_peaks_at_warmup_and_decays() -> None:
    model = torch.nn.Linear(4, 4)
    optimizer = build_optimizer(model, lr=1e-3)
    scheduler = build_scheduler(optimizer, warmup_steps=10)
    lrs = []
    for _ in range(20):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    assert lrs[0] < lrs[9]
    peak = max(lrs)
    assert abs(peak - 1e-3) < 1e-4
    assert lrs[0] < peak
    assert lrs[-1] < peak


def make_data(tmp_path: Path) -> tuple[TokenizerWrapper, Path, Path]:
    tokenizer_path = tmp_path / "tokenizer_shared.json"
    train_tokenizer(
        (text for _ in range(20) for text in CORPUS), vocab_size=400, save_path=tokenizer_path
    )
    wrapper = TokenizerWrapper.from_file(tokenizer_path)
    pairs = [
        TokenizedPair(
            i,
            wrapper.encode(f"hello world {i}"),
            wrapper.encode(f"こんにちは {i}"),
            "synthetic",
        )
        for i in range(12)
    ]
    data_dir = tmp_path / "data"
    train = write_shards(iter(pairs[:8]), data_dir, "train", shard_size=4)
    dev = write_shards(iter(pairs[8:]), data_dir, "jesc-own-dev", shard_size=2)
    manifest = {
        "vocab_size": wrapper.vocab_size,
        "shards": {"train": train.shards, "jesc-own-dev": dev.shards},
        "weights": {"jesc": 0.7, "opus100": 0.3},
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return wrapper, tokenizer_path, data_dir


def test_train_smoke_then_eval(tmp_path: Path) -> None:
    _, tokenizer_path, data_dir = make_data(tmp_path)
    config = TrainConfig(
        direction="en-jp",
        data_dir=str(data_dir),
        tokenizer_path=str(tokenizer_path),
        model_folder=str(tmp_path / "weights"),
        run_name="dev",
        seq_len=16,
        batch_size=4,
        num_epochs=5,
        lr=1e-3,
        warmup_steps=5,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        dropout=0.0,
        num_workers=0,
        val_interval=0,
        val_batches=2,
        amp=False,
        seed=0,
    )
    result = train_model(config, device_name="cpu", max_steps=3)
    assert result["global_step"] == 3

    run_dir = tmp_path / "weights" / "dev"
    assert (run_dir / "weights.safetensors").exists()
    assert (run_dir / "metadata.json").exists()

    _, metadata, _ = load_run_dir(run_dir)
    assert metadata["global_step"] == 3
    assert metadata["direction"] == "en-jp"

    metrics = evaluate(
        run_dir,
        data_dir,
        tokenizer_path=tokenizer_path,
        split="jesc-own-dev",
        device_name="cpu",
        limit=2,
    )
    assert metrics["count"] == 2
    assert set(metrics) >= {"bleu", "chrf", "direction", "split"}


def test_train_config_from_yaml_ignores_unknown(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("direction: jp-en\nbatch_size: 8\nunknown_key: 1\n", encoding="utf-8")
    config = TrainConfig.from_yaml(path)
    assert config.direction == "jp-en"
    assert config.batch_size == 8


def test_mixed_direction_smoke(tmp_path: Path) -> None:
    _, tokenizer_path, data_dir = make_data(tmp_path)
    config = TrainConfig(
        direction="mixed",
        lang_embedding=True,
        data_dir=str(data_dir),
        tokenizer_path=str(tokenizer_path),
        model_folder=str(tmp_path / "weights"),
        run_name="mixed",
        seq_len=16,
        batch_size=4,
        num_epochs=3,
        lr=1e-3,
        warmup_steps=3,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        dropout=0.0,
        num_workers=0,
        val_interval=0,
        val_batches=1,
        amp=False,
        seed=0,
    )
    result = train_model(config, device_name="cpu", max_steps=2)
    assert result["global_step"] == 2
    assert (tmp_path / "weights" / "mixed" / "weights.safetensors").exists()
    _, metadata, _ = load_run_dir(tmp_path / "weights" / "mixed")
    assert metadata["direction"] == "mixed"
    assert metadata["tie_embeddings"] is False


def test_train_writes_and_prunes_snapshots(tmp_path: Path) -> None:
    _, tokenizer_path, data_dir = make_data(tmp_path)
    config = TrainConfig(
        direction="en-jp",
        data_dir=str(data_dir),
        tokenizer_path=str(tokenizer_path),
        model_folder=str(tmp_path / "weights"),
        run_name="snap",
        seq_len=16,
        batch_size=4,
        num_epochs=5,
        lr=1e-3,
        warmup_steps=2,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        dropout=0.0,
        num_workers=0,
        val_interval=0,
        amp=False,
        checkpoint_interval=1,
        snapshot_interval=1,
        keep_checkpoints=2,
        seed=0,
    )
    result = train_model(config, device_name="cpu", max_steps=4)
    assert result["global_step"] == 4
    run_dir = tmp_path / "weights" / "snap"
    assert (run_dir / "weights.safetensors").exists()
    assert (run_dir / "metadata.json").exists()
    snapshots = sorted(path.name for path in run_dir.glob("step-*"))
    assert snapshots == ["step-00000003", "step-00000004"]


def test_train_oom_retries_with_smaller_micro_batch(tmp_path: Path, monkeypatch) -> None:
    import pytorch_transformers.train as train_mod

    _, tokenizer_path, data_dir = make_data(tmp_path)
    original = train_mod._forward_loss
    calls = {"count": 0}

    def flaky_forward_loss(model, batch, loss_fn, use_amp, device):
        calls["count"] += 1
        if calls["count"] == 1:
            raise torch.cuda.OutOfMemoryError("simulated out of memory")
        return original(model, batch, loss_fn, use_amp, device)

    monkeypatch.setattr(train_mod, "_forward_loss", flaky_forward_loss)
    config = TrainConfig(
        direction="en-jp",
        data_dir=str(data_dir),
        tokenizer_path=str(tokenizer_path),
        model_folder=str(tmp_path / "weights"),
        run_name="oom",
        seq_len=16,
        batch_size=4,
        num_epochs=2,
        lr=1e-3,
        warmup_steps=2,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        dropout=0.0,
        num_workers=0,
        val_interval=0,
        amp=False,
        checkpoint_interval=0,
        snapshot_interval=0,
        seed=0,
    )
    result = train_model(config, device_name="cpu", max_steps=1)
    assert result["global_step"] == 1
    assert calls["count"] == 2