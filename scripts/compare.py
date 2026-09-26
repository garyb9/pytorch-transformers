from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from pytorch_transformers.checkpoints import load_model
from pytorch_transformers.tokenizer import TokenizerWrapper
from pytorch_transformers.translate import translate_text
from pytorch_transformers.utils import resolve_device

ROOT = Path(__file__).resolve().parents[1]
RUST_DIR = ROOT
RUST_BIN = ROOT / "target" / "release" / "pytorch-transformers-rs"

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


def bench_python(args, texts: list[str]) -> tuple[list[str], list[float]]:
    device = resolve_device(args.device)
    model, _ = load_model(args.python_model)
    model.to(device).eval()
    tokenizer = TokenizerWrapper.from_file(args.tokenizer)

    def run() -> list[str]:
        return [
            translate_text(model, tokenizer, text, device, max_len=args.max_len, beam=args.beam)
            for text in texts
        ]

    for _ in range(args.warmup):
        run()
    times: list[float] = []
    outputs: list[str] = []
    for _ in range(args.reps):
        start = time.perf_counter()
        outputs = run()
        times.append(time.perf_counter() - start)
    return outputs, times


def ensure_rust_binary() -> float:
    if RUST_BIN.exists():
        return 0.0
    start = time.perf_counter()
    subprocess.run(["cargo", "build", "--release"], cwd=RUST_DIR, check=True)
    return time.perf_counter() - start


def bench_rust(args, texts: list[str]) -> tuple[dict[str, Any], float]:
    build_seconds = ensure_rust_binary()
    command = [
        str(RUST_BIN),
        "bench",
        "--model",
        str(args.rust_model),
        "--config",
        str(args.rust_config),
        "--tokenizer",
        str(args.tokenizer),
        "--max-len",
        str(args.max_len),
        "--reps",
        str(args.reps),
        "--warmup",
        str(args.warmup),
        "--device",
        args.device,
    ]
    for text in texts:
        command.extend(["--text", text])
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    payload: dict[str, Any] = json.loads(process.stdout.strip().splitlines()[-1])
    return payload, build_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Python and Rust inference.")
    fixture = ROOT / "crates" / "transformer" / "tests" / "fixtures" / "parity_small"
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
    rust_payload, build_seconds = bench_rust(args, texts)
    rust_outputs = rust_payload["outputs"]

    parity = py_outputs == rust_outputs
    results: dict[str, Any] = {
        "device": args.device,
        "beam": args.beam,
        "reps": args.reps,
        "warmup": args.warmup,
        "texts": texts,
        "python": {
            "median_s": percentile(py_times, 0.5),
            "p95_s": percentile(py_times, 0.95),
            "min_s": min(py_times),
            "outputs": py_outputs,
        },
        "rust": {
            "median_s": rust_payload["median_s"],
            "p95_s": rust_payload["p95_s"],
            "min_s": rust_payload["min_s"],
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