from __future__ import annotations

import pytest

from pytorch_transformers import ModelConfig


def test_from_dict_ignores_unknown_keys() -> None:
    config = ModelConfig.from_dict(
        {"src_vocab_size": 10, "tgt_vocab_size": 10, "batch_size": 8, "nonsense": True}
    )
    assert config.src_vocab_size == 10
    assert config.tgt_vocab_size == 10


def test_to_dict_roundtrip() -> None:
    config = ModelConfig(src_vocab_size=10, tgt_vocab_size=12, d_model=32, n_heads=4)
    assert ModelConfig.from_dict(config.to_dict()) == config


def test_invalid_residual_mode_rejected() -> None:
    with pytest.raises(ValueError, match="residual_mode"):
        ModelConfig(src_vocab_size=10, tgt_vocab_size=10, residual_mode="bogus")


def test_d_model_divisibility_enforced() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(src_vocab_size=10, tgt_vocab_size=10, d_model=10, n_heads=4)