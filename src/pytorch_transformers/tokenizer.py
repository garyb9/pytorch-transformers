from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "<2en>", "<2ja>"]
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
LANG_EN_TOKEN = "<2en>"
LANG_JA_TOKEN = "<2ja>"


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int = 32000,
    save_path: str | Path | None = None,
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    if save_path is not None:
        save_tokenizer(tokenizer, save_path)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(target))


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TokenizerWrapper:
    def __init__(self, tokenizer: Tokenizer, path: str | Path | None = None) -> None:
        self.tokenizer = tokenizer
        self.path = Path(path) if path is not None else None

    @classmethod
    def from_file(cls, path: str | Path) -> TokenizerWrapper:
        return cls(load_tokenizer(path), path)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def token_id(self, token: str) -> int:
        index = self.tokenizer.token_to_id(token)
        if index is None:
            raise KeyError(f"token not found in vocabulary: {token!r}")
        return index

    @property
    def pad_id(self) -> int:
        return self.token_id(PAD_TOKEN)

    @property
    def unk_id(self) -> int:
        return self.token_id(UNK_TOKEN)

    @property
    def bos_id(self) -> int:
        return self.token_id(BOS_TOKEN)

    @property
    def eos_id(self) -> int:
        return self.token_id(EOS_TOKEN)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def sha256(self) -> str | None:
        return file_sha256(self.path) if self.path is not None else None