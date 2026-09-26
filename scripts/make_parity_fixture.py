from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from pytorch_transformers.checkpoints import export_safetensors
from pytorch_transformers.config import ModelConfig
from pytorch_transformers.model import build_transformer
from pytorch_transformers.tokenizer import TokenizerWrapper, train_tokenizer
from pytorch_transformers.translate import beam_search, greedy_decode

CORPUS = [
    "hello world",
    "the quick brown fox jumps",
    "こんにちは世界",
    "日本語のテストです",
    "attention is all you need",
    "機械翻訳の学習",
]


def build_fixture(
    out: Path,
    config: ModelConfig,
    direction: str,
    langs: tuple[int | None, int | None],
) -> None:
    torch.manual_seed(0)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer_path = out / "tokenizer.json"
    train_tokenizer(
        (text for _ in range(40) for text in CORPUS), vocab_size=400, save_path=tokenizer_path
    )
    wrapper = TokenizerWrapper.from_file(tokenizer_path)
    vocab_size = wrapper.vocab_size

    config = ModelConfig.from_dict(
        {**config.to_dict(), "src_vocab_size": vocab_size, "tgt_vocab_size": vocab_size}
    )
    model = build_transformer(config).eval()
    export_safetensors(model, out / "model.safetensors", config, direction)

    pad = wrapper.pad_id
    src = torch.randint(1, vocab_size, (2, 9))
    src[:, -1] = pad
    tgt = torch.randint(1, vocab_size, (2, 7))
    tgt[:, -1] = pad

    src_mask = (src != pad).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt != pad).unsqueeze(1).unsqueeze(2) & torch.tril(
        torch.ones(1, 1, tgt.size(1), tgt.size(1), dtype=torch.bool)
    )

    src_lang_id, tgt_lang_id = langs
    src_lang = None if src_lang_id is None else torch.full((2,), src_lang_id, dtype=torch.long)
    tgt_lang = None if tgt_lang_id is None else torch.full((2,), tgt_lang_id, dtype=torch.long)

    with torch.no_grad():
        logits = model(src, tgt, src_mask, tgt_mask, src_lang, tgt_lang)
    save_file({"logits": logits.contiguous()}, str(out / "expected.safetensors"))

    source_ids = [5, 6, 7, 8, 9]
    expected_greedy = greedy_decode(
        model,
        wrapper,
        source_ids,
        torch.device("cpu"),
        max_len=8,
        src_lang=src_lang_id,
        tgt_lang=tgt_lang_id,
    )
    expected_beam = beam_search(
        model,
        wrapper,
        source_ids,
        torch.device("cpu"),
        max_len=8,
        beam_size=3,
        length_penalty=0.6,
        src_lang=src_lang_id,
        tgt_lang=tgt_lang_id,
    )

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
                "bos_id": wrapper.bos_id,
                "eos_id": wrapper.eos_id,
                "src_lang_id": src_lang_id,
                "tgt_lang_id": tgt_lang_id,
                "expected_greedy": expected_greedy,
                "expected_beam": expected_beam,
            }
        ),
        encoding="utf-8",
    )
    print(f"wrote fixture to {out} (vocab {vocab_size}, lang_embedding={config.lang_embedding})")


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "crates" / "transformer" / "tests" / "fixtures"
    small = ModelConfig(
        src_vocab_size=1,
        tgt_vocab_size=1,
        src_seq_len=64,
        tgt_seq_len=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        residual_mode="post",
    )
    build_fixture(root / "parity_small", small, "en-jp", (None, None))
    mixed = ModelConfig.from_dict({**small.to_dict(), "lang_embedding": True})
    build_fixture(root / "parity_mixed", mixed, "mixed", (0, 1))


if __name__ == "__main__":
    main()