from __future__ import annotations

import json

import pytest
import torch

from pytorch_transformers.dataset import (
    DIRECTIONS,
    LANG_IDS,
    TranslationDataset,
    causal_mask,
)
from pytorch_transformers.tokenizer import TokenizerWrapper, train_tokenizer

CORPUS = ["hello world", "the quick brown fox", "こんにちは世界", "日本語のテスト"]


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory) -> TokenizerWrapper:
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    train_tokenizer((text for _ in range(50) for text in CORPUS), vocab_size=400, save_path=path)
    return TokenizerWrapper.from_file(path)


def write_shard(path, tokenizer: TokenizerWrapper, pairs) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, (src, tgt) in enumerate(pairs):
            record = {
                "src": tokenizer.encode(src),
                "tgt": tokenizer.encode(tgt),
                "origin": "test",
                "pair_id": index,
            }
            handle.write(json.dumps(record) + "\n")


def test_causal_mask_blocks_future() -> None:
    mask = causal_mask(4)
    assert mask.shape == (4, 4)
    assert not mask[0, 1].item()
    assert mask[1, 0].item()
    assert mask[3, 3].item()


def test_item_shapes_and_lengths(tmp_path, tokenizer: TokenizerWrapper) -> None:
    shard = tmp_path / "train-00000.jsonl"
    write_shard(shard, tokenizer, [("hello world", "こんにちは世界")])
    dataset = TranslationDataset([shard], tokenizer, seq_len=16)
    item = dataset[0]
    assert len(dataset) == 1
    for key in ("encoder_input", "decoder_input", "label"):
        assert item[key].shape == (16,)
    assert item["encoder_mask"].shape == (1, 1, 16)
    assert item["decoder_mask"].shape == (1, 16, 16)
    assert item["encoder_mask"].dtype == torch.bool
    assert item["decoder_mask"].dtype == torch.bool


def test_padding_and_special_tokens(tmp_path, tokenizer: TokenizerWrapper) -> None:
    shard = tmp_path / "train-00000.jsonl"
    write_shard(shard, tokenizer, [("hello world", "こんにちは世界")])
    item = TranslationDataset([shard], tokenizer, seq_len=16)[0]
    pad = tokenizer.pad_id
    assert item["encoder_input"][0].item() == tokenizer.bos_id
    assert item["decoder_input"][0].item() == tokenizer.bos_id
    assert item["encoder_input"][-1].item() == pad
    src_body_len = len(tokenizer.encode("hello world"))
    assert item["encoder_input"][src_body_len + 1].item() == tokenizer.eos_id


def test_direction_swaps_encoder_and_decoder(tmp_path, tokenizer: TokenizerWrapper) -> None:
    shard = tmp_path / "train-00000.jsonl"
    write_shard(shard, tokenizer, [("hello world", "こんにちは世界")])
    forward = TranslationDataset([shard], tokenizer, seq_len=16, direction="en-jp")[0]
    backward = TranslationDataset([shard], tokenizer, seq_len=16, direction="jp-en")[0]
    assert forward["encoder_input"][1].item() == tokenizer.encode("hello world")[0]
    assert backward["encoder_input"][1].item() == tokenizer.encode("こんにちは世界")[0]


def test_invalid_direction_rejected(tmp_path, tokenizer: TokenizerWrapper) -> None:
    shard = tmp_path / "train-00000.jsonl"
    write_shard(shard, tokenizer, [("hello world", "こんにちは世界")])
    with pytest.raises(ValueError, match="direction"):
        TranslationDataset([shard], tokenizer, seq_len=16, direction="xx-yy")


def test_directions_constant() -> None:
    assert DIRECTIONS == ("en-jp", "jp-en", "mixed")


def test_lang_ids_reflect_direction(tmp_path, tokenizer: TokenizerWrapper) -> None:
    shard = tmp_path / "train-00000.jsonl"
    write_shard(shard, tokenizer, [("hello world", "こんにちは世界")])
    forward = TranslationDataset([shard], tokenizer, seq_len=16, direction="en-jp")[0]
    backward = TranslationDataset([shard], tokenizer, seq_len=16, direction="jp-en")[0]
    assert forward["src_lang_id"].item() == LANG_IDS["en"]
    assert forward["tgt_lang_id"].item() == LANG_IDS["jp"]
    assert backward["src_lang_id"].item() == LANG_IDS["jp"]
    assert backward["tgt_lang_id"].item() == LANG_IDS["en"]


def test_mixed_direction_uses_both_orientations(tmp_path, tokenizer: TokenizerWrapper) -> None:
    shard = tmp_path / "train-00000.jsonl"
    write_shard(shard, tokenizer, [(f"hello world {i}", f"こんにちは {i}") for i in range(12)])
    dataset = TranslationDataset([shard], tokenizer, seq_len=16, direction="mixed")
    source_langs = {item["src_lang_id"].item() for item in dataset}
    assert source_langs == {LANG_IDS["en"], LANG_IDS["jp"]}