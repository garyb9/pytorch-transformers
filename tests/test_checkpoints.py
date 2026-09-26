from __future__ import annotations

import torch

from pytorch_transformers.checkpoints import (
    expected_keys,
    export_safetensors,
    load_model,
    load_training_checkpoint,
    save_training_checkpoint,
)
from pytorch_transformers.config import ModelConfig
from pytorch_transformers.model import build_transformer


def small_config(**overrides) -> ModelConfig:
    values = {
        "src_vocab_size": 20,
        "tgt_vocab_size": 20,
        "src_seq_len": 8,
        "tgt_seq_len": 8,
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 32,
        "dropout": 0.0,
    }
    values.update(overrides)
    return ModelConfig(**values)


def random_batch(config: ModelConfig):
    src = torch.randint(0, config.src_vocab_size, (2, config.src_seq_len))
    tgt = torch.randint(0, config.tgt_vocab_size, (2, config.tgt_seq_len))
    src_mask = torch.ones(2, 1, 1, config.src_seq_len, dtype=torch.bool)
    tgt_mask = torch.tril(torch.ones(1, config.tgt_seq_len, config.tgt_seq_len, dtype=torch.bool))
    return src, tgt, src_mask, tgt_mask


def test_export_keys_match_contract_untied(tmp_path) -> None:
    config = small_config()
    model = build_transformer(config)
    payload = export_safetensors(model, tmp_path / "model.safetensors", config, "en-jp")
    from safetensors.torch import load_file

    saved = set(load_file(str(tmp_path / "model.safetensors")))
    assert saved == expected_keys(config)
    assert payload["direction"] == "en-jp"
    assert payload["format"] == 1


def test_export_and_reload_produces_same_logits(tmp_path) -> None:
    config = small_config()
    model = build_transformer(config).eval()
    path = tmp_path / "model.safetensors"
    export_safetensors(model, path, config, "en-jp")
    reloaded, payload = load_model(path, direction="en-jp")
    reloaded.eval()
    assert payload["direction"] == "en-jp"

    src, tgt, src_mask, tgt_mask = random_batch(config)
    with torch.no_grad():
        expected = model(src, tgt, src_mask, tgt_mask)
        actual = reloaded(src, tgt, src_mask, tgt_mask)
    assert torch.allclose(expected, actual, atol=1e-5)


def test_tied_embeddings_export_and_reload(tmp_path) -> None:
    config = small_config(tie_embeddings=True)
    model = build_transformer(config).eval()
    path = tmp_path / "tied.safetensors"
    export_safetensors(model, path, config, "en-jp")

    from safetensors.torch import load_file

    saved = set(load_file(str(path)))
    assert saved == expected_keys(config)
    assert "tgt_embed.weight" not in saved
    assert "tgt_proj.weight" not in saved

    reloaded, _ = load_model(path)
    reloaded.eval()
    src, tgt, src_mask, tgt_mask = random_batch(config)
    with torch.no_grad():
        expected = model(src, tgt, src_mask, tgt_mask)
        actual = reloaded(src, tgt, src_mask, tgt_mask)
    assert torch.allclose(expected, actual)


def test_lang_embedding_export_roundtrip(tmp_path) -> None:
    config = small_config(lang_embedding=True)
    model = build_transformer(config).eval()
    path = tmp_path / "mixed.safetensors"
    export_safetensors(model, path, config, "mixed")

    from safetensors.torch import load_file

    saved = set(load_file(str(path)))
    assert saved == expected_keys(config)
    assert "lang_embed.weight" in saved

    reloaded, _ = load_model(path)
    reloaded.eval()
    src, tgt, src_mask, tgt_mask = random_batch(config)
    src_lang = torch.zeros(2, dtype=torch.long)
    tgt_lang = torch.ones(2, dtype=torch.long)
    with torch.no_grad():
        expected = model(src, tgt, src_mask, tgt_mask, src_lang, tgt_lang)
        actual = reloaded(src, tgt, src_mask, tgt_mask, src_lang, tgt_lang)
    assert torch.allclose(expected, actual)


def test_training_checkpoint_roundtrip(tmp_path) -> None:
    config = small_config()
    model = build_transformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    run_dir = tmp_path / "run"
    metadata = {
        "epoch": 3,
        "global_step": 42,
        "direction": "jp-en",
        "tie_embeddings": False,
        "tokenizer_sha256": "abc",
    }
    save_training_checkpoint(run_dir, model, optimizer, metadata)

    fresh = build_transformer(config)
    fresh_optimizer = torch.optim.Adam(fresh.parameters(), lr=1e-3)
    loaded = load_training_checkpoint(run_dir, fresh, fresh_optimizer)
    assert loaded == metadata


def test_direction_mismatch_is_rejected(tmp_path) -> None:
    config = small_config()
    model = build_transformer(config)
    path = tmp_path / "model.safetensors"
    export_safetensors(model, path, config, "en-jp")
    try:
        load_model(path, direction="jp-en")
    except ValueError as error:
        assert "direction" in str(error)
    else:
        raise AssertionError("expected ValueError for direction mismatch")