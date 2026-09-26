from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .tokenizer import TokenizerWrapper

DIRECTIONS = ("en-jp", "jp-en", "mixed")
LANG_IDS = {"en": 0, "jp": 1}


def causal_mask(size: int) -> torch.Tensor:
    return torch.tril(torch.ones(size, size, dtype=torch.bool))


def mixed_orientation(pair_id: int, seed: int) -> bool:
    digest = hashlib.sha1(f"{seed}:{pair_id}".encode()).digest()
    return digest[0] % 2 == 0


class TranslationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        shards: Sequence[str | Path],
        tokenizer: TokenizerWrapper,
        seq_len: int,
        direction: str = "en-jp",
        seed: int = 42,
    ) -> None:
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
        self.shards = [Path(shard) for shard in shards]
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.direction = direction
        self.seed = seed
        self._index: list[tuple[int, int]] = []
        for shard_index, path in enumerate(self.shards):
            with path.open("rb") as handle:
                offset = 0
                for line in handle:
                    self._index.append((shard_index, offset))
                    offset += len(line)

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for index in range(len(self)):
            yield self[index]

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
        if self.direction == "jp-en":
            src_ids, tgt_ids = tgt_ids, src_ids
            src_lang, tgt_lang = LANG_IDS["jp"], LANG_IDS["en"]
        elif self.direction == "mixed":
            en_to_jp = mixed_orientation(int(record.get("pair_id", index)), self.seed)
            if not en_to_jp:
                src_ids, tgt_ids = tgt_ids, src_ids
            src_lang = LANG_IDS["en"] if en_to_jp else LANG_IDS["jp"]
            tgt_lang = LANG_IDS["jp"] if en_to_jp else LANG_IDS["en"]
        else:
            src_lang, tgt_lang = LANG_IDS["en"], LANG_IDS["jp"]

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
            "src_lang_id": torch.tensor(src_lang, dtype=torch.long),
            "tgt_lang_id": torch.tensor(tgt_lang, dtype=torch.long),
        }