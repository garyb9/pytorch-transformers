from __future__ import annotations

from pytorch_transformers.metrics import corpus_bleu, corpus_chrf


def test_identical_sequences_score_max_bleu() -> None:
    sentences = ["the cat sat on the mat", "a quick brown fox jumps high"]
    assert corpus_bleu(sentences, sentences) > 99.9


def test_chrf_high_for_identical_sequences() -> None:
    assert corpus_chrf(["the cat sat"], ["the cat sat"]) > 90.0


def test_metrics_handle_multiple_sentences() -> None:
    hypotheses = ["hello world how are you", "the quick brown fox jumps"]
    references = ["hello world how are you", "the quick brown fox jumps"]
    assert corpus_bleu(hypotheses, references) > 99.9
    assert corpus_chrf(hypotheses, references) > 90.0