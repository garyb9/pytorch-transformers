from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .tokenizer import TokenizerWrapper


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def ids_hash(ids: Sequence[int]) -> str:
    payload = ",".join(str(index) for index in ids)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def pair_hash(src_ids: Sequence[int], tgt_ids: Sequence[int]) -> str:
    payload = f"{ids_hash(src_ids)}|{ids_hash(tgt_ids)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class Pair:
    pair_id: int
    src: str
    tgt: str
    origin: str


@dataclass(slots=True)
class TokenizedPair:
    pair_id: int
    src_ids: list[int]
    tgt_ids: list[int]
    origin: str


@dataclass(slots=True)
class SourceSpec:
    id: str
    hub_id: str
    split: str
    src_field: str
    tgt_field: str
    weight: float
    config: str | None = None
    max_examples: int | None = None
    validation_split: str | None = None
    test_split: str | None = None
    license: str | None = None
    attribution: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceSpec:
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass(slots=True)
class FilterConfig:
    seq_len: int = 256
    max_length_ratio: float = 3.0


@dataclass(slots=True)
class DataConfig:
    sources: list[SourceSpec]
    seed: int = 42
    filters: FilterConfig = field(default_factory=FilterConfig)
    tokenizer_vocab_size: int = 32000
    tokenizer_sample_size: int = 4_000_000
    own_dev_fraction: float = 0.0007
    own_test_fraction: float = 0.0007
    canonical_src_lang: str = "en"
    canonical_tgt_lang: str = "jp"

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        tokenizer_raw = raw.get("tokenizer", {})
        own_eval_raw = raw.get("own_eval", {})
        return cls(
            sources=[SourceSpec.from_dict(item) for item in raw.get("sources", [])],
            seed=raw.get("seed", 42),
            filters=FilterConfig(**raw.get("filters", {})),
            tokenizer_vocab_size=tokenizer_raw.get("vocab_size", 32000),
            tokenizer_sample_size=tokenizer_raw.get("sample_size", 4_000_000),
            own_dev_fraction=own_eval_raw.get("dev_fraction", 0.0007),
            own_test_fraction=own_eval_raw.get("test_fraction", 0.0007),
        )

    def source(self, source_id: str) -> SourceSpec:
        for spec in self.sources:
            if spec.id == source_id:
                return spec
        raise KeyError(f"unknown source: {source_id!r}")


def resolve_field(row: dict[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        value = value[part]
    return value


def load_hf_pairs(
    spec: SourceSpec, split: str | None = None, limit: int | None = None
) -> Iterator[Pair]:
    from datasets import load_dataset

    target_split = split or spec.split
    if spec.config:
        dataset = load_dataset(spec.hub_id, spec.config, split=target_split)
    else:
        dataset = load_dataset(spec.hub_id, split=target_split)
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        src = str(resolve_field(row, spec.src_field))
        tgt = str(resolve_field(row, spec.tgt_field))
        yield Pair(index, src, tgt, spec.id)


def tokenize_and_filter(
    pairs: Iterable[Pair],
    tokenizer: TokenizerWrapper,
    config: FilterConfig,
) -> Iterator[TokenizedPair]:
    limit = config.seq_len - 2
    for pair in pairs:
        src_ids = tokenizer.encode(normalize(pair.src))
        tgt_ids = tokenizer.encode(normalize(pair.tgt))
        if not src_ids or not tgt_ids:
            continue
        if len(src_ids) > limit or len(tgt_ids) > limit:
            continue
        smaller = min(len(src_ids), len(tgt_ids))
        if max(len(src_ids), len(tgt_ids)) / smaller > config.max_length_ratio:
            continue
        yield TokenizedPair(pair.pair_id, src_ids, tgt_ids, pair.origin)


def dedup(pairs: Iterable[TokenizedPair]) -> Iterator[TokenizedPair]:
    seen: set[str] = set()
    for pair in pairs:
        key = pair_hash(pair.src_ids, pair.tgt_ids)
        if key in seen:
            continue
        seen.add(key)
        yield pair


def text_blocked_hashes(texts: Iterable[str], tokenizer: TokenizerWrapper) -> set[str]:
    blocked: set[str] = set()
    for text in texts:
        cleaned = normalize(text)
        if cleaned:
            blocked.add(ids_hash(tokenizer.encode(cleaned)))
    return blocked


def pair_text_hashes(pairs: Iterable[Pair], tokenizer: TokenizerWrapper) -> Iterator[str]:
    for pair in pairs:
        for text in (pair.src, pair.tgt):
            cleaned = normalize(text)
            if cleaned:
                yield ids_hash(tokenizer.encode(cleaned))


def read_tsv_pairs(path: str | Path, origin: str = "jesc-official") -> Iterator[Pair]:
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                yield Pair(index, parts[0], parts[1], origin)


def decontaminate(pairs: Iterable[TokenizedPair], blocked: set[str]) -> Iterator[TokenizedPair]:
    for pair in pairs:
        if ids_hash(pair.src_ids) in blocked or ids_hash(pair.tgt_ids) in blocked:
            continue
        yield pair


def bucket_for(
    origin: str,
    pair_id: int,
    seed: int,
    dev_fraction: float,
    test_fraction: float,
) -> str:
    digest = hashlib.sha1(f"{seed}:{origin}:{pair_id}".encode()).digest()
    value = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    if value < test_fraction:
        return "test"
    if value < test_fraction + dev_fraction:
        return "dev"
    return "train"


def interleave(
    streams: Sequence[tuple[Iterable[TokenizedPair], float]],
    seed: int,
) -> Iterator[TokenizedPair]:
    rng = random.Random(seed)
    iterators = [iter(stream) for stream, _ in streams]
    weights = [weight for _, weight in streams]
    while iterators:
        index = rng.choices(range(len(iterators)), weights=weights)[0]
        try:
            yield next(iterators[index])
        except StopIteration:
            iterators.pop(index)
            weights.pop(index)


@dataclass(slots=True)
class ShardResult:
    shards: list[str]
    count: int
    max_src_len: int
    max_tgt_len: int


def write_shards(
    pairs: Iterable[TokenizedPair],
    out_dir: str | Path,
    prefix: str,
    shard_size: int = 100_000,
) -> ShardResult:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    shards: list[str] = []
    count = 0
    max_src_len = 0
    max_tgt_len = 0
    buffer: list[TokenizedPair] = []

    def flush(index: int) -> None:
        path = directory / f"{prefix}-{index:05d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for item in buffer:
                record = {
                    "src": item.src_ids,
                    "tgt": item.tgt_ids,
                    "origin": item.origin,
                    "pair_id": item.pair_id,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        shards.append(path.name)
        buffer.clear()

    for pair in pairs:
        buffer.append(pair)
        count += 1
        max_src_len = max(max_src_len, len(pair.src_ids))
        max_tgt_len = max(max_tgt_len, len(pair.tgt_ids))
        if len(buffer) >= shard_size:
            flush(len(shards))
    if buffer:
        flush(len(shards))

    return ShardResult(shards, count, max_src_len, max_tgt_len)


def read_shard(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_data(
    config: DataConfig,
    tokenizer_path: str | Path,
    out_dir: str | Path,
    max_examples: int | None = None,
    jesc_official_dir: str | Path | None = None,
    shard_size: int = 100_000,
) -> dict[str, Any]:
    tokenizer = TokenizerWrapper.from_file(tokenizer_path)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    def bounded(spec: SourceSpec) -> int | None:
        limits = [value for value in (spec.max_examples, max_examples) if value is not None]
        return min(limits) if limits else None

    blocked: set[str] = set()
    stats: dict[str, Any] = {"sources": {}, "dropped": {}}

    opus = config.source("opus100")
    for split in (opus.validation_split, opus.test_split):
        if split:
            blocked |= set(pair_text_hashes(load_hf_pairs(opus, split=split), tokenizer))

    jesc = config.source("jesc")
    own_dev: list[TokenizedPair] = []
    own_test: list[TokenizedPair] = []

    def jesc_bucket(pair: Pair) -> str:
        return bucket_for(
            jesc.id,
            pair.pair_id,
            config.seed,
            config.own_dev_fraction,
            config.own_test_fraction,
        )

    for pair in load_hf_pairs(jesc, limit=bounded(jesc)):
        bucket = jesc_bucket(pair)
        if bucket == "train":
            continue
        for tokenized in tokenize_and_filter([pair], tokenizer, config.filters):
            if bucket == "dev":
                own_dev.append(tokenized)
            else:
                own_test.append(tokenized)
            blocked.add(ids_hash(tokenized.src_ids))
            blocked.add(ids_hash(tokenized.tgt_ids))
    stats["sources"]["jesc"] = {"own_dev": len(own_dev), "own_test": len(own_test)}

    if jesc_official_dir is not None:
        official = Path(jesc_official_dir)
        for name in ("dev", "test"):
            candidate = official / f"{name}.tsv"
            if candidate.exists():
                blocked |= set(pair_text_hashes(read_tsv_pairs(candidate), tokenizer))

    def jesc_train_pairs() -> Iterator[Pair]:
        for pair in load_hf_pairs(jesc, limit=bounded(jesc)):
            if jesc_bucket(pair) == "train":
                yield pair

    opus_train = tokenize_and_filter(
        load_hf_pairs(opus, limit=bounded(opus)), tokenizer, config.filters
    )
    opus_clean = decontaminate(opus_train, blocked)
    jesc_train = decontaminate(
        tokenize_and_filter(jesc_train_pairs(), tokenizer, config.filters), blocked
    )

    mixed = interleave([(opus_clean, opus.weight), (jesc_train, jesc.weight)], config.seed)
    deduped = dedup(mixed)
    if max_examples is not None:
        deduped = (pair for index, pair in enumerate(deduped) if index < max_examples)

    train_result = write_shards(deduped, directory, "train", shard_size)
    dev_result = write_shards(iter(own_dev), directory, "jesc-own-dev", shard_size)
    test_result = write_shards(iter(own_test), directory, "jesc-own-test", shard_size)

    manifest = {
        "version": 1,
        "seed": config.seed,
        "canonical": {"src": config.canonical_src_lang, "tgt": config.canonical_tgt_lang},
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": tokenizer.sha256(),
        "weights": {spec.id: spec.weight for spec in config.sources},
        "filters": {
            "seq_len": config.filters.seq_len,
            "max_length_ratio": config.filters.max_length_ratio,
        },
        "counts": {
            "train": train_result.count,
            "jesc-own-dev": dev_result.count,
            "jesc-own-test": test_result.count,
        },
        "max_lens": {
            "train_src": train_result.max_src_len,
            "train_tgt": train_result.max_tgt_len,
        },
        "shards": {
            "train": train_result.shards,
            "jesc-own-dev": dev_result.shards,
            "jesc-own-test": test_result.shards,
        },
        "sources": [
            {
                "id": spec.id,
                "hub_id": spec.hub_id,
                "license": spec.license,
                "attribution": spec.attribution,
            }
            for spec in config.sources
        ],
    }
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "stats.json", stats)
    return manifest