from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer

from .checkpoints import export_safetensors, load_run_dir
from .data import DataConfig, prepare_data, resolve_field
from .eval import evaluate
from .tokenizer import TokenizerWrapper, train_tokenizer
from .train import TrainConfig, train_model
from .translate import translate_text
from .utils import resolve_device

app = typer.Typer(add_completion=False, help="Transformer playground for EN<->JP translation.")


@app.command("train-tokenizer")
def train_tokenizer_command(
    data_config: Path = typer.Argument(..., exists=True),
    out: Path = typer.Option(Path("tokenizer_shared.json"), "--out"),
    vocab_size: int | None = typer.Option(None, "--vocab-size"),
    sample: int = typer.Option(2_000_000, "--sample"),
) -> None:
    from datasets import load_dataset

    config = DataConfig.from_yaml(data_config)
    vocab = vocab_size or config.tokenizer_vocab_size
    per_source = max(1, sample // max(1, len(config.sources)))

    def texts():
        for spec in config.sources:
            if spec.config:
                dataset = load_dataset(spec.hub_id, spec.config, split=spec.split, streaming=True)
            else:
                dataset = load_dataset(spec.hub_id, split=spec.split, streaming=True)
            for index, row in enumerate(dataset):
                if index >= per_source:
                    break
                yield str(resolve_field(row, spec.src_field))
                yield str(resolve_field(row, spec.tgt_field))

    tokenizer = train_tokenizer(texts(), vocab_size=vocab, save_path=out)
    typer.echo(f"tokenizer with {tokenizer.get_vocab_size()} tokens written to {out}")


@app.command("prepare-data")
def prepare_data_command(
    data_config: Path = typer.Argument(..., exists=True),
    tokenizer_path: Path = typer.Option(Path("tokenizer_shared.json"), "--tokenizer"),
    out: Path = typer.Option(Path("data/opus_jesc"), "--out"),
    max_examples: int | None = typer.Option(None, "--max-examples"),
    jesc_official: Path | None = typer.Option(None, "--jesc-official"),
) -> None:
    config = DataConfig.from_yaml(data_config)
    manifest = prepare_data(
        config,
        tokenizer_path,
        out,
        max_examples=max_examples,
        jesc_official_dir=jesc_official,
    )
    typer.echo(json.dumps(manifest["counts"], indent=2))


@app.command("train")
def train_command(
    config_path: Path = typer.Argument(..., exists=True),
    device: str = typer.Option("auto", "--device"),
    resume: str | None = typer.Option(None, "--resume"),
    max_steps: int | None = typer.Option(None, "--max-steps"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = TrainConfig.from_yaml(config_path)
    if resume is not None:
        config.resume = resume
    result = train_model(config, device_name=device, max_steps=max_steps)
    typer.echo(json.dumps(result, indent=2))


@app.command("eval")
def eval_command(
    run_dir: Path = typer.Argument(..., exists=True),
    data_dir: Path = typer.Option(Path("data/opus_jesc"), "--data-dir"),
    tokenizer_path: Path | None = typer.Option(None, "--tokenizer"),
    split: str | None = typer.Option(None, "--split"),
    device: str = typer.Option("auto", "--device"),
    beam: int = typer.Option(1, "--beam"),
    limit: int | None = typer.Option(None, "--limit"),
) -> None:
    result = evaluate(
        run_dir,
        data_dir,
        tokenizer_path=tokenizer_path,
        split=split,
        device_name=device,
        beam=beam,
        limit=limit,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("translate")
def translate_command(
    run_dir: Path = typer.Argument(..., exists=True),
    text: str | None = typer.Option(None, "--text"),
    file: Path | None = typer.Option(None, "--file"),
    tokenizer_path: Path | None = typer.Option(None, "--tokenizer"),
    device: str = typer.Option("auto", "--device"),
    beam: int = typer.Option(1, "--beam"),
    max_len: int = typer.Option(256, "--max-len"),
) -> None:
    model, metadata, _ = load_run_dir(run_dir)
    tokenizer_file = tokenizer_path or metadata.get("tokenizer_path") or "tokenizer_shared.json"
    tokenizer = TokenizerWrapper.from_file(tokenizer_file)
    resolved = resolve_device(device)
    model.to(resolved).eval()

    if file is not None:
        lines = file.read_text(encoding="utf-8").splitlines()
    elif text is not None:
        lines = [text]
    else:
        lines = [line for line in sys.stdin.read().splitlines() if line.strip()]

    for line in lines:
        if line.strip():
            typer.echo(translate_text(model, tokenizer, line, resolved, max_len=max_len, beam=beam))


@app.command("export")
def export_command(
    run_dir: Path = typer.Argument(..., exists=True),
    out: Path = typer.Option(Path("model.safetensors"), "--out"),
) -> None:
    model, metadata, config = load_run_dir(run_dir)
    tokenizer_path = metadata.get("tokenizer_path") or "tokenizer_shared.json"
    tokenizer_sha = metadata.get("tokenizer_sha256")
    if Path(tokenizer_path).exists():
        tokenizer_sha = TokenizerWrapper.from_file(tokenizer_path).sha256()
    export_safetensors(model, out, config, str(metadata["direction"]), tokenizer_sha)
    typer.echo(f"exported {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()