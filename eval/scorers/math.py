"""Math scorer — exact match on the FINAL numeric answer (spec §4).

We score only the final number, not the reasoning chain. Extraction tolerates
formatting: '$', thousands commas, trailing period, and the gsm8k '#### N' marker.
Comparison is numeric (so '42', '42.0', '42.00' all match) with a tiny tolerance.

Per-item: {"correct": 0/1, "pred_num": float|None, "ref_num": float|None, "reason": str}
Corpus:   {"accuracy", "unparsable_pred_rate", "n"}
"""

from __future__ import annotations

import re

from ._util import clean

# Signed integers/decimals, optionally with thousands separators, e.g. -1,234.50
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _to_float(token: str) -> float | None:
    token = token.replace(",", "").rstrip(".").strip()
    try:
        return float(token)
    except ValueError:
        return None


def extract_final_number(text: str) -> float | None:
    """Return the final numeric answer from `text`, or None if none found.

    Priority:
      1. number following the gsm8k '####' marker
      2. number following an 'answer' cue ('answer is', 'the answer:', '= ')
      3. the LAST number appearing in the text
    """
    if text is None:
        return None
    t = clean(text)
    t = t.replace("$", "").replace("%", "")

    # 1. '#### 42'
    m = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", t)
    if m:
        return _to_float(m.group(1))

    # 2. answer cue
    m = None
    for mm in re.finditer(r"(?:answer(?:\s+is)?\s*[:=]?\s*|=\s*)(-?\d[\d,]*(?:\.\d+)?)",
                          t, flags=re.IGNORECASE):
        m = mm  # take the last such cue
    if m:
        return _to_float(m.group(1))

    # 3. last number anywhere
    nums = _NUM_RE.findall(t)
    if nums:
        return _to_float(nums[-1])
    return None


def score_item(pred: str, ref: str, extra: dict | None = None) -> dict:
    pred_num = extract_final_number(pred)
    # ref may be a raw gsm8k answer ("... #### 18") or a bare number.
    ref_num = extract_final_number(ref)

    if ref_num is None:
        return {"correct": 0, "pred_num": pred_num, "ref_num": None, "reason": "ref_unparsable"}
    if pred_num is None:
        return {"correct": 0, "pred_num": None, "ref_num": ref_num, "reason": "pred_unparsable"}

    correct = int(abs(pred_num - ref_num) < 1e-6)
    return {"correct": correct, "pred_num": pred_num, "ref_num": ref_num,
            "reason": "match" if correct else "wrong_number"}


def score_corpus(items: list[dict]) -> dict:
    n = len(items)
    if n == 0:
        return {"accuracy": 0.0, "unparsable_pred_rate": 0.0, "n": 0}
    correct = sum(it["correct"] for it in items)
    unparsable = sum(1 for it in items if it["pred_num"] is None)
    return {"accuracy": correct / n, "unparsable_pred_rate": unparsable / n, "n": n}
