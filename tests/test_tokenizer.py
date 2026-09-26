from __future__ import annotations

import pytest

from pytorch_transformers.tokenizer import (
    TokenizerWrapper,
    file_sha256,
    load_tokenizer,
    train_tokenizer,
)

CORPUS = [
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "transformers are attention based models",
    "attention is all you need",
    "こんにちは世界",
    "日本語のテストです",
    "機械翻訳の学習",
    "東京は日本の首都です",
]


@pytest.fixture(scope="module")
def wrapper(tmp_path_factory) -> TokenizerWrapper:
    path = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    train_tokenizer((text for _ in range(50) for text in CORPUS), vocab_size=400, save_path=path)
    return TokenizerWrapper.from_file(path)


def test_special_token_ids_are_fixed(wrapper: TokenizerWrapper) -> None:
    assert wrapper.pad_id == 0
    assert wrapper.unk_id == 1
    assert wrapper.bos_id == 2
    assert wrapper.eos_id == 3


def test_roundtrip_english(wrapper: TokenizerWrapper) -> None:
    text = "hello world"
    assert wrapper.decode(wrapper.encode(text)) == text


def test_roundtrip_japanese(wrapper: TokenizerWrapper) -> None:
    text = "こんにちは世界"
    assert wrapper.decode(wrapper.encode(text)) == text


def test_japanese_has_no_unknown_tokens(wrapper: TokenizerWrapper) -> None:
    assert wrapper.unk_id not in wrapper.encode("日本語のテストです")


def test_save_load_and_hash_match(tmp_path) -> None:
    path = tmp_path / "tokenizer.json"
    train_tokenizer((text for _ in range(50) for text in CORPUS), vocab_size=400, save_path=path)
    reloaded = TokenizerWrapper.from_file(path)
    assert reloaded.sha256() == file_sha256(path)
    assert reloaded.vocab_size == load_tokenizer(path).get_vocab_size()