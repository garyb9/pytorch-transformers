from __future__ import annotations

import torch

from pytorch_transformers.config import ModelConfig
from pytorch_transformers.model import build_transformer
from pytorch_transformers.translate import (
    beam_search,
    content_ids,
    greedy_decode,
    translate_text,
)


class StubTokenizer:
    bos_id = 2
    eos_id = 3
    pad_id = 0

    def encode(self, text: str) -> list[int]:
        return [7, 8]

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(index) for index in ids)


def build_model() -> torch.nn.Module:
    config = ModelConfig(
        src_vocab_size=20,
        tgt_vocab_size=20,
        src_seq_len=16,
        tgt_seq_len=16,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        dropout=0.0,
    )
    return build_transformer(config).eval()


def test_greedy_stays_within_max_len() -> None:
    model = build_model()
    output = greedy_decode(model, StubTokenizer(), [7, 8], torch.device("cpu"), max_len=5)
    assert isinstance(output, list)
    assert 0 <= len(output) <= 6


def test_beam_returns_tokens() -> None:
    model = build_model()
    output = beam_search(
        model, StubTokenizer(), [7, 8], torch.device("cpu"), max_len=5, beam_size=3
    )
    assert isinstance(output, list)
    assert 0 <= len(output) <= 6


def test_translate_text_returns_string() -> None:
    model = build_model()
    text = translate_text(model, StubTokenizer(), "hello", torch.device("cpu"), max_len=5)
    assert isinstance(text, str)


def test_content_ids_strips_special_tokens() -> None:
    assert content_ids([2, 7, 8, 3, 0, 0], StubTokenizer()) == [7, 8]