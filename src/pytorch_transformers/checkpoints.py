from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .config import ModelConfig
from .model import Transformer


def expected_keys(config: ModelConfig) -> set[str]:
    keys: set[str] = {
        "src_embed.weight",
        "encoder.norm.weight",
        "encoder.norm.bias",
        "decoder.norm.weight",
        "decoder.norm.bias",
        "tgt_proj.bias",
    }
    if not config.tie_embeddings:
        keys |= {"tgt_embed.weight", "tgt_proj.weight"}
    for index in range(config.n_layers):
        for prefix in (f"encoder.layers.{index}",):
            for name in ("w_q", "w_k", "w_v", "w_o"):
                keys |= {f"{prefix}.self_attn.{name}.weight", f"{prefix}.self_attn.{name}.bias"}
            keys |= {f"{prefix}.ffn.linear1.weight", f"{prefix}.ffn.linear1.bias"}
            keys |= {f"{prefix}.ffn.linear2.weight", f"{prefix}.ffn.linear2.bias"}
            keys |= {f"{prefix}.norm1.weight", f"{prefix}.norm1.bias"}
            keys |= {f"{prefix}.norm2.weight", f"{prefix}.norm2.bias"}

        prefix = f"decoder.layers.{index}"
        for attention in ("self_attn", "cross_attn"):
            for name in ("w_q", "w_k", "w_v", "w_o"):
                keys |= {
                    f"{prefix}.{attention}.{name}.weight",
                    f"{prefix}.{attention}.{name}.bias",
                }
        keys |= {f"{prefix}.ffn.linear1.weight", f"{prefix}.ffn.linear1.bias"}
        keys |= {f"{prefix}.ffn.linear2.weight", f"{prefix}.ffn.linear2.bias"}
        keys |= {f"{prefix}.norm1.weight", f"{prefix}.norm1.bias"}
        keys |= {f"{prefix}.norm2.weight", f"{prefix}.norm2.bias"}
        keys |= {f"{prefix}.norm3.weight", f"{prefix}.norm3.bias"}
    return keys


def model_state_dict(model: Transformer, tie_embeddings: bool) -> dict[str, torch.Tensor]:
    state = {key: value for key, value in model.state_dict().items()}
    if tie_embeddings:
        state.pop("tgt_embed.weight", None)
        state.pop("tgt_proj.weight", None)
    return state


def load_into_model(
    model: Transformer, state: Mapping[str, torch.Tensor], tie_embeddings: bool
) -> None:
    weights = dict(state)
    if tie_embeddings:
        weights["tgt_embed.weight"] = weights["src_embed.weight"]
        weights["tgt_proj.weight"] = weights["src_embed.weight"]
    model.load_state_dict(weights, strict=True)


def export_safetensors(
    model: Transformer,
    path: str | Path,
    config: ModelConfig,
    direction: str,
    tokenizer_sha256: str | None = None,
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = model_state_dict(model, config.tie_embeddings)
    save_file(state, str(target))

    payload: dict[str, Any] = {
        "format": 1,
        "direction": direction,
        "tokenizer_sha256": tokenizer_sha256,
        **config.to_dict(),
    }
    target.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def load_model(
    path: str | Path, direction: str | None = None
) -> tuple[Transformer, dict[str, Any]]:
    from .model import build_transformer

    weights_path = Path(path)
    payload = json.loads(weights_path.with_suffix(".json").read_text(encoding="utf-8"))
    if direction is not None and payload.get("direction") != direction:
        raise ValueError(
            f"checkpoint direction {payload.get('direction')!r} "
            f"does not match requested {direction!r}"
        )
    config = ModelConfig.from_dict(payload)
    config.tie_embeddings = bool(payload.get("tie_embeddings", False))
    model = build_transformer(config)
    state = load_file(str(weights_path))
    load_into_model(model, state, config.tie_embeddings)
    return model, payload


def save_training_checkpoint(
    run_dir: str | Path,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
) -> Path:
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tie_embeddings = bool(metadata.get("tie_embeddings", False))
    save_file(model_state_dict(model, tie_embeddings), str(directory / "weights.safetensors"))
    torch.save(optimizer.state_dict(), directory / "optimizer.pt")
    (directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return directory


def load_training_checkpoint(
    run_dir: str | Path,
    model: Transformer,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    directory = Path(run_dir)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    state = load_file(str(directory / "weights.safetensors"))
    load_into_model(model, state, bool(metadata.get("tie_embeddings", False)))
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(directory / "optimizer.pt", map_location="cpu"))
    return metadata


def load_run_dir(run_dir: str | Path) -> tuple[Transformer, dict[str, Any], ModelConfig]:
    from .model import build_transformer

    directory = Path(run_dir)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    config = ModelConfig.from_dict(metadata["model_config"])
    config.tie_embeddings = bool(metadata.get("tie_embeddings", False))
    model = build_transformer(config)
    state = load_file(str(directory / "weights.safetensors"))
    load_into_model(model, state, config.tie_embeddings)
    return model, metadata, config