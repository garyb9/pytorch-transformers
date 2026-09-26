from __future__ import annotations

from collections.abc import Sequence

import sacrebleu


def corpus_bleu(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    return float(sacrebleu.corpus_bleu(list(hypotheses), [list(references)]).score)


def corpus_chrf(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    return float(sacrebleu.corpus_chrf(list(hypotheses), [list(references)]).score)