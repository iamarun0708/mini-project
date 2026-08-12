"""Translation scorer — chrF++ via sacrebleu, corpus-level (spec §4).

chrF++ (character n-grams + word bigrams, i.e. word_order=2) is more robust than
BLEU at this scale. We report the CORPUS-level score. This scorer is inherently
corpus-level, so `score_item` records the cleaned hypothesis/reference and the
real number comes from `score_corpus`.

Per-item: {"hyp": str, "ref": str}
Corpus:   {"chrf++": float, "n": int}
"""

from __future__ import annotations

import sacrebleu

from ._util import clean


def _clean_hyp(text: str) -> str:
    t = clean(text)
    # Strip a leading "Translation:" / "English:" label if the model adds one.
    for prefix in ("translation:", "english:", "english translation:"):
        if t.lower().startswith(prefix):
            t = t[len(prefix):].strip()
    return t


def score_item(pred: str, ref: str, extra: dict | None = None) -> dict:
    return {"hyp": _clean_hyp(pred), "ref": clean(ref)}


def score_corpus(items: list[dict]) -> dict:
    n = len(items)
    if n == 0:
        return {"chrf++": 0.0, "n": 0}
    hyps = [it["hyp"] for it in items]
    refs = [[it["ref"] for it in items]]  # sacrebleu: list of reference streams
    # word_order=2 => chrF++ ; char_order=6, beta=2 are sacrebleu defaults.
    chrf = sacrebleu.corpus_chrf(hyps, refs, word_order=2)
    return {"chrf++": chrf.score, "n": n}
