from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    src_seq_len: int = 256
    tgt_seq_len: int = 256
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    residual_mode: str = "post"
    tie_embeddings: bool = False
    lang_embedding: bool = False

    def __post_init__(self) -> None:
        if self.residual_mode not in {"post", "pre"}:
            raise ValueError(f"residual_mode must be 'post' or 'pre', got {self.residual_mode!r}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}