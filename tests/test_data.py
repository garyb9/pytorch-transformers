from __future__ import annotations

from pathlib import Path

from pytorch_transformers.data import (
    DataConfig,
    FilterConfig,
    Pair,
    SourceSpec,
    TokenizedPair,
    bucket_for,
    decontaminate,
    dedup,
    interleave,
    load_csv_pairs,
    normalize,
    prepare_data,
    read_shard,
    source_available,
    text_blocked_hashes,
    tokenize_and_filter,
    write_shards,
)
from pytorch_transformers.tokenizer import train_tokenizer


class CharTokenizer:
    pad_id = 0
    unk_id = 1

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(index) for index in ids)


def test_normalize_fullwidth_and_spaces() -> None:
    assert normalize("  ＡＢＣ   de　f ") == "ABC de f"


def test_tokenize_and_filter_drops_long_and_skewed() -> None:
    tokenizer = CharTokenizer()
    config = FilterConfig(seq_len=6, max_length_ratio=3.0)
    pairs = [
        Pair(0, "abcd", "ab", "s"),
        Pair(1, "abcdefgh", "ab", "s"),
        Pair(2, "a", "abcd", "s"),
        Pair(3, "abc", "abcd", "s"),
    ]
    kept = {pair.pair_id for pair in tokenize_and_filter(pairs, tokenizer, config)}
    assert kept == {0, 3}


def test_dedup_removes_duplicate_pairs() -> None:
    pairs = [
        TokenizedPair(0, [1, 2], [3, 4], "s"),
        TokenizedPair(1, [1, 2], [3, 4], "s"),
        TokenizedPair(2, [1, 2], [5, 6], "s"),
    ]
    assert [pair.pair_id for pair in dedup(pairs)] == [0, 2]


def test_decontaminate_drops_eval_overlap() -> None:
    tokenizer = CharTokenizer()
    blocked = text_blocked_hashes(["abc"], tokenizer)
    pairs = [
        TokenizedPair(0, [ord("a"), ord("b"), ord("c")], [100], "s"),
        TokenizedPair(1, [200], [201], "s"),
    ]
    assert [pair.pair_id for pair in decontaminate(pairs, blocked)] == [1]


def test_bucket_for_is_deterministic_and_balanced() -> None:
    first = bucket_for("jesc", 123, 42, 0.1, 0.1)
    assert first == bucket_for("jesc", 123, 42, 0.1, 0.1)
    counts = {"train": 0, "dev": 0, "test": 0}
    for pair_id in range(10000):
        counts[bucket_for("jesc", pair_id, 42, 0.1, 0.1)] += 1
    assert 850 < counts["dev"] < 1150
    assert 850 < counts["test"] < 1150
    assert 7500 < counts["train"] < 8500


def test_interleave_includes_all_streams_deterministically() -> None:
    left = [TokenizedPair(i, [i], [i], "left") for i in range(4)]
    right = [TokenizedPair(i, [i], [i], "right") for i in range(4)]

    def run() -> list[str]:
        return [pair.origin for pair in interleave([(left, 0.7), (right, 0.3)], seed=1)]

    result = run()
    assert result == run()
    assert result.count("left") == 4
    assert result.count("right") == 4


def test_write_shards_and_read_back(tmp_path) -> None:
    pairs = [TokenizedPair(i, [i, i + 1], [i + 2], "s") for i in range(5)]
    result = write_shards(iter(pairs), tmp_path, "train", shard_size=2)
    assert result.count == 5
    assert len(result.shards) == 3
    assert result.max_src_len == 2
    assert result.max_tgt_len == 1
    records = read_shard(tmp_path / result.shards[0])
    assert len(records) == 2
    assert set(records[0]) == {"src", "tgt", "origin", "pair_id"}

def test_prepare_data_end_to_end_with_fake_loader(tmp_path, monkeypatch) -> None:
    corpus = [f"hello world {i}" for i in range(40)] + [f"こんにちは {i}" for i in range(40)]
    tokenizer_path = tmp_path / "tokenizer.json"
    train_tokenizer(
        (text for _ in range(20) for text in corpus), vocab_size=400, save_path=tokenizer_path
    )

    def fake_loader(spec, split=None, limit=None):
        target = split or spec.split
        offset = 0 if spec.id == "opus100" else 10000
        if spec.id == "opus100" and target in ("validation", "test"):
            yield Pair(offset, "eval only source", "評価専用", spec.id)
            return
        count = limit if limit is not None else 40
        for index in range(count):
            yield Pair(offset + index, f"hello world {index}", f"こんにちは {index}", spec.id)

    monkeypatch.setattr("pytorch_transformers.data.load_hf_pairs", fake_loader)

    csv_path = tmp_path / "jlpt.csv"
    lines = ["english,japanese,level"]
    for index in range(20):
        lines.append(f"study sentence {index},勉強 {index},N{index % 5 + 1}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sources = [
        SourceSpec(
            id="jesc",
            hub_id="fake/jesc",
            split="train",
            src_field="translation.en",
            tgt_field="translation.ja",
            weight=0.7,
        ),
        SourceSpec(
            id="opus100",
            hub_id="fake/opus",
            config="en-ja",
            split="train",
            validation_split="validation",
            test_split="test",
            src_field="translation.en",
            tgt_field="translation.ja",
            weight=0.3,
        ),
        SourceSpec(
            id="jlpt",
            format="csv",
            path=str(csv_path),
            src_field="english",
            tgt_field="japanese",
            level_field="level",
            weight=0.1,
            eval_fraction=0.5,
        ),
    ]
    config = DataConfig(
        sources=sources,
        seed=7,
        filters=FilterConfig(seq_len=32, max_length_ratio=3.0),
        own_dev_fraction=0.25,
        own_test_fraction=0.25,
    )

    out_dir = tmp_path / "prepared"
    manifest = prepare_data(config, tokenizer_path, out_dir, max_examples=60, shard_size=8)

    assert manifest["counts"]["train"] > 0
    assert manifest["counts"]["jlpt-eval"] > 0
    assert manifest["tokenizer_sha256"] is not None
    assert manifest["weights"] == {"jesc": 0.7, "opus100": 0.3, "jlpt": 0.1}
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "stats.json").exists()
    assert Path(out_dir / manifest["shards"]["train"][0]).exists()
    assert manifest["sources"][1]["id"] == "opus100"
    eval_records = read_shard(out_dir / manifest["shards"]["jlpt-eval"][0])
    assert all("level" in record for record in eval_records)


def test_load_csv_pairs_parses_fields_and_levels(tmp_path) -> None:
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "english,japanese,level\nHello,こんにちは,N5\nThanks,ありがとう,N4\n",
        encoding="utf-8",
    )
    spec = SourceSpec(
        id="jlpt",
        format="csv",
        path=str(csv_path),
        src_field="english",
        tgt_field="japanese",
        level_field="level",
        weight=1.0,
    )
    pairs = list(load_csv_pairs(spec))
    assert [(pair.src, pair.tgt, pair.level) for pair in pairs] == [
        ("Hello", "こんにちは", "N5"),
        ("Thanks", "ありがとう", "N4"),
    ]
    assert source_available(spec)
    missing = SourceSpec(
        id="missing",
        format="csv",
        path=str(tmp_path / "nope.csv"),
        src_field="a",
        tgt_field="b",
        weight=1.0,
    )
    assert not source_available(missing)
