from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoints import load_run_dir
from .dataset import TranslationDataset
from .metrics import corpus_bleu, corpus_chrf
from .tokenizer import TokenizerWrapper
from .translate import content_ids, translate_ids
from .utils import resolve_device


def evaluate(
    run_dir: str | Path,
    data_dir: str | Path,
    tokenizer_path: str | Path | None = None,
    split: str | None = None,
    device_name: str = "auto",
    beam: int = 1,
    limit: int | None = None,
) -> dict[str, Any]:
    model, metadata, model_config = load_run_dir(run_dir)
    direction = str(metadata.get("direction", "en-ja"))
    tokenizer_file = tokenizer_path or metadata.get("tokenizer_path") or "tokenizer_shared.json"
    tokenizer = TokenizerWrapper.from_file(tokenizer_file)

    manifest = json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))
    if split is None:
        split = "jesc-own-test" if manifest["shards"].get("jesc-own-test") else "train"
    shards = [Path(data_dir) / name for name in manifest["shards"][split]]

    seq_len = model_config.src_seq_len
    dataset = TranslationDataset(shards, tokenizer, seq_len, direction=direction)
    device = resolve_device(device_name)
    model.to(device).eval()

    hypotheses: list[str] = []
    references: list[str] = []
    for index, item in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        source_ids = content_ids(item["encoder_input"].tolist(), tokenizer)
        predicted = translate_ids(model, tokenizer, source_ids, device, seq_len, beam=beam)
        hypotheses.append(tokenizer.decode(predicted))
        references.append(tokenizer.decode(content_ids(item["label"].tolist(), tokenizer)))

    result: dict[str, Any] = {
        "split": split,
        "direction": direction,
        "beam": beam,
        "count": len(hypotheses),
    }
    if hypotheses:
        result["bleu"] = corpus_bleu(hypotheses, references)
        result["chrf"] = corpus_chrf(hypotheses, references)
    else:
        result["bleu"] = 0.0
        result["chrf"] = 0.0
    return result