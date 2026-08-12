"""JSON extraction scorer — schema validity + per-field F1 (spec §4).

Two numbers, both reported:
  1. schema_valid_rate: fraction of outputs that parse AND validate against the schema.
  2. field_f1: micro-averaged per-field F1 over the VALID outputs, comparing predicted
     (key, value) pairs to the gold object.

Design note (spec): "a model that outputs valid-but-empty JSON must not score well."
F1 enforces this — an empty {} that validates (all fields optional) gets recall 0,
hence F1 0. So a high schema_valid_rate with low field_f1 correctly signals a model
gaming the schema.

`extra` carries the per-item schema and gold object:
    extra = {"schema": <jsonschema dict>, "target": <gold dict>}

Per-item: {"schema_valid": bool, "parse_ok": bool, "tp": int, "fp": int, "fn": int}
Corpus:   {"schema_valid_rate", "field_f1", "field_precision", "field_recall", "n"}
"""

from __future__ import annotations

import json

from jsonschema import Draft7Validator

from ._util import clean


def _parse(text: str):
    """Best-effort JSON parse. Returns (obj, ok). Extracts the first {...} span if
    the model wraps the object in prose."""
    t = clean(text)
    try:
        return json.loads(t), True
    except Exception:
        pass
    # Fallback: grab the outermost brace span.
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(t[start:end + 1]), True
        except Exception:
            return None, False
    return None, False


def _norm_value(v) -> str:
    """Canonical string form of a scalar/collection value for comparison."""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        # 3 and 3.0 compare equal
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    if isinstance(v, str):
        return " ".join(v.lower().split())
    if isinstance(v, (list, dict)):
        return json.dumps(v, sort_keys=True, ensure_ascii=False).lower()
    if v is None:
        return "null"
    return str(v).lower()


def _field_counts(pred: dict, gold: dict) -> tuple[int, int, int]:
    """(tp, fp, fn) over (key,value) pairs. A predicted key counts as TP only if the
    key exists in gold AND the normalized values match."""
    tp = fp = 0
    for k, v in pred.items():
        if k in gold and _norm_value(v) == _norm_value(gold[k]):
            tp += 1
        else:
            fp += 1
    fn = 0
    for k, v in gold.items():
        if not (k in pred and _norm_value(pred[k]) == _norm_value(v)):
            fn += 1
    return tp, fp, fn


def score_item(pred: str, ref: str, extra: dict | None = None) -> dict:
    extra = extra or {}
    schema = extra.get("schema")
    gold = extra.get("target")
    if gold is None and ref:
        try:
            gold = json.loads(ref)
        except Exception:
            gold = {}
    gold = gold or {}

    obj, parse_ok = _parse(pred)
    if not parse_ok or not isinstance(obj, dict):
        return {"schema_valid": False, "parse_ok": parse_ok, "tp": 0, "fp": 0,
                "fn": len(gold)}

    schema_valid = True
    if schema is not None:
        schema_valid = Draft7Validator(schema).is_valid(obj)

    tp, fp, fn = _field_counts(obj, gold)
    return {"schema_valid": bool(schema_valid), "parse_ok": True,
            "tp": tp, "fp": fp, "fn": fn}


def score_corpus(items: list[dict]) -> dict:
    n = len(items)
    if n == 0:
        return {"schema_valid_rate": 0.0, "field_f1": 0.0, "field_precision": 0.0,
                "field_recall": 0.0, "n": 0}
    valid = sum(1 for it in items if it["schema_valid"])
    # Micro F1 over VALID items only (spec: F1 "over the valid ones").
    tp = sum(it["tp"] for it in items if it["schema_valid"])
    fp = sum(it["fp"] for it in items if it["schema_valid"])
    fn = sum(it["fn"] for it in items if it["schema_valid"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "schema_valid_rate": valid / n,
        "field_f1": f1,
        "field_precision": precision,
        "field_recall": recall,
        "n": n,
    }
