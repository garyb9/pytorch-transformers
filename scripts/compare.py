from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from pytorch_transformers.checkpoints import load_model
from pytorch_transformers.tokenizer import TokenizerWrapper
from pytorch_transformers.translate import translate_text
from pytorch_transformers.utils import resolve_device

ROOT = Path(__file__).resolve().parents[1]
RUST_DIR = ROOT / "rust"
RUST_BIN = RUST_DIR / "target" / "release" / "pytorch-transformers-rs"

DEFAULT_TEXTS = [
    "hello world",
    "attention is all you need",
    "こんにちは世界",
    "機械翻訳の学習",
]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def python_translate(args, texts: list[str]) -> list[str]:
    device = resolve_device(args.device)
    model, _ = load_model(args.python_model)
    model.to(device).eval()
    tokenizer = TokenizerWrapper.from_file(args.tokenizer)
    return [
        translate_text(model, tokenizer, text, device, max_len=args.max_len, beam=args.beam)
        for text in texts
    ]


def bench_python(args, texts: list[str]) -> tuple[list[str], list[float]]:
    for _ in range(args.warmup):
        python_translate(args, texts)
    times: list[float] = []
    outputs: list[str] = []
    for _ in range(args.reps):
        start = time.perf_counter()
        outputs = python_translate(args, texts)
        times.append(time.perf_counter() - start)
    return outputs, times


def ensure_rust_binary() -> float:
    if RUST_BIN.exists():
        return 0.0
    start = time.perf_counter()
    subprocess.run(["cargo", "build", "--release"], cwd=RUST_DIR, check=True)
    return time.perf_counter() - start


def rust_translate(args, texts: list[str], input_file: str) -> list[str]:
    command = [
        str(RUST_BIN),
        "translate",
        "--model",
        str(args.rust_model),
        "--config",
        str(args.rust_config),
        "--tokenizer",
        str(args.tokenizer),
        "--file",
        input_file,
        "--max-len",
        str(args.max_len),
        "--device",
        args.device,
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    return process.stdout.strip().splitlines()


def bench_rust(args, texts: list[str]) -> tuple[list[str], list[float], float]:
    build_seconds = ensure_rust_binary()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        handle.write("\n".join(texts) + "\n")
        input_file = handle.name
    try:
        for _ in range(args.warmup):
            rust_translate(args, texts, input_file)
        times: list[float] = []
        outputs: list[str] = []
        for _ in range(args.reps):
            start = time.perf_counter()
            outputs = rust_translate(args, texts, input_file)
            times.append(time.perf_counter() - start)
    finally:
        Path(input_file).unlink(missing_ok=True)
    return outputs, times, build_seconds


def summarise(times: list[float]) -> dict[str, float]:
    return {
        "median_s": percentile(times, 0.5),
        "p95_s": percentile(times, 0.95),
        "min_s": min(times),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Python and Rust inference.")
    fixture = ROOT / "rust" / "tests" / "fixtures" / "parity_small"
    parser.add_argument("--python-model", default=str(fixture / "model.safetensors"))
    parser.add_argument("--rust-model", default=str(fixture / "model.safetensors"))
    parser.add_argument("--rust-config", default=str(fixture / "model.json"))
    parser.add_argument("--tokenizer", default=str(fixture / "tokenizer.json"))
    parser.add_argument("--text", action="append", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", default=str(ROOT / "bench" / "results.json"))
    args = parser.parse_args()

    texts = args.text or DEFAULT_TEXTS
    torch.set_grad_enabled(False)

    py_outputs, py_times = bench_python(args, texts)
    rust_outputs, rust_times, build_seconds = bench_rust(args, texts)

    parity = py_outputs == rust_outputs
    results: dict[str, Any] = {
        "device": args.device,
        "beam": args.beam,
        "reps": args.reps,
        "texts": texts,
        "python": {**summarise(py_times), "outputs": py_outputs},
        "rust": {
            **summarise(rust_times),
            "outputs": rust_outputs,
            "build_s": build_seconds,
        },
        "outputs_match": parity,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"device={args.device} reps={args.reps} outputs_match={parity}")
    print(f"{'stack':<8}{'median_s':>12}{'p95_s':>12}{'min_s':>12}")
    for name in ("python", "rust"):
        row = results[name]
        print(f"{name:<8}{row['median_s']:>12.4f}{row['p95_s']:>12.4f}{row['min_s']:>12.4f}")
    print(f"results written to {out_path}")


if __name__ == "__main__":
    main()