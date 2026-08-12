"""SQL scorer — AST equivalence via sqlglot (spec §4).

NOT string exact match. We parse predicted and reference SQL, normalize
(lowercase identifiers, sort top-level SELECT columns where order is irrelevant),
then compare canonical serializations. Parse failures are logged SEPARATELY from
wrong answers so we can tell "the model produced non-SQL" from "the model produced
the wrong query".

Per-item result dict:
    {"correct": 0/1, "parse_ok_pred": bool, "parse_ok_ref": bool, "reason": str}
Corpus aggregate:
    {"accuracy", "parse_failure_rate", "n"}
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from ._util import clean


def _normalize(sql: str) -> str | None:
    """Return a canonical SQL string, or None if it does not parse."""
    try:
        tree = sqlglot.parse_one(sql)
    except Exception:
        return None
    if tree is None:
        return None

    # Lowercase all identifiers (table/column/alias names).
    for node in tree.walk():
        if isinstance(node, exp.Identifier):
            this = node.args.get("this")
            if isinstance(this, str):
                node.set("this", this.lower())

    # Sort top-level SELECT projection columns — output column order is irrelevant
    # to whether the query answers the question (spec: "sort where order is irrelevant").
    # We deliberately do NOT reorder ORDER BY / GROUP BY, where order is meaningful.
    select = tree.find(exp.Select)
    if select is not None and select.args.get("expressions"):
        exprs = list(select.expressions)
        select.set("expressions", sorted(exprs, key=lambda e: e.sql(normalize=True)))

    # Canonical serialization: uppercase keywords, normalized spacing.
    try:
        return tree.sql(normalize=True, pretty=False).lower()
    except Exception:
        return None


def score_item(pred: str, ref: str, extra: dict | None = None) -> dict:
    pred_clean = clean(pred).rstrip(";").strip()
    ref_clean = clean(ref).rstrip(";").strip()

    npred = _normalize(pred_clean)
    nref = _normalize(ref_clean)

    parse_ok_pred = npred is not None
    parse_ok_ref = nref is not None

    if not parse_ok_ref:
        # Reference itself failed to parse — a data problem, not a model error.
        return {"correct": 0, "parse_ok_pred": parse_ok_pred, "parse_ok_ref": False,
                "reason": "ref_parse_failed"}
    if not parse_ok_pred:
        return {"correct": 0, "parse_ok_pred": False, "parse_ok_ref": True,
                "reason": "pred_parse_failed"}

    correct = int(npred == nref)
    return {"correct": correct, "parse_ok_pred": True, "parse_ok_ref": True,
            "reason": "match" if correct else "ast_mismatch"}


def score_corpus(items: list[dict]) -> dict:
    n = len(items)
    if n == 0:
        return {"accuracy": 0.0, "parse_failure_rate": 0.0, "n": 0}
    correct = sum(it["correct"] for it in items)
    pred_parse_failures = sum(1 for it in items if not it["parse_ok_pred"])
    return {
        "accuracy": correct / n,
        "parse_failure_rate": pred_parse_failures / n,
        "n": n,
    }
