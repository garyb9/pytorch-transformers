from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from pytorch_transformers.checkpoints import export_safetensors
from pytorch_transformers.config import ModelConfig
from pytorch_transformers.model import build_transformer
from pytorch_transformers.tokenizer import TokenizerWrapper, train_tokenizer
from pytorch_transformers.translate import greedy_decode

CORPUS = [
    "hello world",
    "the quick brown fox jumps",
    "こんにちは世界",
    "日本語のテストです",
    "attention is all you need",
    "機械翻訳の学習",
]


def main() -> None:
    torch.manual_seed(0)
    out = Path("rust/tests/fixtures/parity_small")
    out.mkdir(parents=True, exist_ok=True)

    config = ModelConfig(
        src_vocab_size=40,
        tgt_vocab_size=40,
        src_seq_len=16,
        tgt_seq_len=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        residual_mode="post",
    )
    model = build_transformer(config).eval()
    export_safetensors(model, out / "model.safetensors", config, "en-ja")

    pad = 0
    src = torch.randint(1, 40, (2, 9))
    src[:, -1] = pad
    tgt = torch.randint(1, 40, (2, 7))
    tgt[:, -1] = pad

    src_mask = (src != pad).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt != pad).unsqueeze(1).unsqueeze(2) & torch.tril(
        torch.ones(1, 1, tgt.size(1), tgt.size(1), dtype=torch.bool)
    )

    with torch.no_grad():
        logits = model(src, tgt, src_mask, tgt_mask)

    save_file({"logits": logits.contiguous()}, str(out / "expected.safetensors"))

    class Stub:
        bos_id = 2
        eos_id = 3
        pad_id = 0

    source_ids = [5, 6, 7, 8, 9]
    expected_greedy = greedy_decode(model, Stub(), source_ids, torch.device("cpu"), max_len=8)

    tokenizer_path = out / "tokenizer.json"
    train_tokenizer(
        (text for _ in range(40) for text in CORPUS), vocab_size=400, save_path=tokenizer_path
    )
    wrapper = TokenizerWrapper.from_file(tokenizer_path)
    cases = ["hello world", "こんにちは世界", "attention is all you need"]
    (out / "tokenizer_cases.json").write_text(
        json.dumps({"texts": cases, "ids": [wrapper.encode(text) for text in cases]}),
        encoding="utf-8",
    )

    (out / "inputs.json").write_text(
        json.dumps(
            {
                "src": src.tolist(),
                "tgt": tgt.tolist(),
                "pad_id": pad,
                "source_ids": source_ids,
                "max_len": 8,
                "bos_id": 2,
                "eos_id": 3,
                "expected_greedy": expected_greedy,
            }
        ),
        encoding="utf-8",
    )
    print(f"wrote fixture to {out}")


if __name__ == "__main__":
    main()