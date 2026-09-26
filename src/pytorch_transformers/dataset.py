from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .tokenizer import TokenizerWrapper

DIRECTIONS = ("en-ja", "ja-en")


def causal_mask(size: int) -> torch.Tensor:
    return torch.tril(torch.ones(size, size, dtype=torch.bool))


class TranslationDataset(Dataset):
    def __init__(
        self,
        shards: Sequence[str | Path],
        tokenizer: TokenizerWrapper,
        seq_len: int,
        direction: str = "en-ja",
    ) -> None:
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
        self.shards = [Path(shard) for shard in shards]
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.direction = direction
        self._index: list[tuple[int, int]] = []
        for shard_index, path in enumerate(self.shards):
            with path.open("rb") as handle:
                offset = 0
                for line in handle:
                    self._index.append((shard_index, offset))
                    offset += len(line)

    def __len__(self) -> int:
        return len(self._index)

    def _load(self, index: int) -> dict[str, Any]:
        shard_index, offset = self._index[index]
        with self.shards[shard_index].open("rb") as handle:
            handle.seek(offset)
            line = handle.readline()
        record: dict[str, Any] = json.loads(line.decode("utf-8"))
        return record

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self._load(index)
        src_ids = list(record["src"])
        tgt_ids = list(record["tgt"])
        if self.direction == "ja-en":
            src_ids, tgt_ids = tgt_ids, src_ids

        pad = self.tokenizer.pad_id
        bos = self.tokenizer.bos_id
        eos = self.tokenizer.eos_id
        body = self.seq_len - 2
        src_body = src_ids[:body]
        tgt_body = tgt_ids[:body]

        encoder_input = torch.tensor(
            [bos, *src_body, eos, *([pad] * (self.seq_len - len(src_body) - 2))],
            dtype=torch.long,
        )
        decoder_input = torch.tensor(
            [bos, *tgt_body, *([pad] * (self.seq_len - len(tgt_body) - 1))],
            dtype=torch.long,
        )
        label = torch.tensor(
            [*tgt_body, eos, *([pad] * (self.seq_len - len(tgt_body) - 1))],
            dtype=torch.long,
        )

        encoder_mask = encoder_input.ne(pad).view(1, 1, self.seq_len)
        decoder_mask = decoder_input.ne(pad).view(1, self.seq_len, 1) & causal_mask(
            self.seq_len
        ).view(1, self.seq_len, self.seq_len)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
        }