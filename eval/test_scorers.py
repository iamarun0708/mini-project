"""Scorer verification gate (spec §11 step 3).

Each scorer is exercised against KNOWN-GOOD and KNOWN-BAD outputs, plus the tricky
cases each scorer is specifically supposed to handle (formatting tolerance, parse
failures, semantic-equivalence, valid-but-empty JSON). If the scorers are wrong,
everything downstream is noise — so this must pass before training anything.

Run:  .venv/bin/python -m eval.test_scorers
Exits non-zero if any check fails.
"""

from __future__ import annotations

import sys

from eval.scorers import sql, math as math_scorer, translation, json_extract

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
def test_sql():
    print("SQL scorer")
    # known-good: exact
    r = sql.score_item("SELECT name FROM users WHERE age > 30",
                       "SELECT name FROM users WHERE age > 30")
    check("sql exact match -> correct", r["correct"] == 1)

    # semantic equivalence: identifier case + column order differences
    r = sql.score_item("select AGE, NAME from Users where AGE > 30",
                       "SELECT name, age FROM users WHERE age > 30")
    check("sql case+col-order equivalent -> correct", r["correct"] == 1, str(r))

    # tolerate markdown fence + trailing semicolon
    r = sql.score_item("```sql\nSELECT name FROM users WHERE age > 30;\n```",
                       "SELECT name FROM users WHERE age > 30")
    check("sql fenced+semicolon -> correct", r["correct"] == 1, str(r))

    # known-bad: genuinely different query
    r = sql.score_item("SELECT name FROM users WHERE age < 30",
                       "SELECT name FROM users WHERE age > 30")
    check("sql wrong predicate -> incorrect", r["correct"] == 0)

    # parse failure is distinguished from wrong answer
    r = sql.score_item("this is not sql at all !!!",
                       "SELECT name FROM users WHERE age > 30")
    check("sql non-sql -> pred_parse_failed",
          r["correct"] == 0 and r["reason"] == "pred_parse_failed", str(r))

    # corpus: parse failures logged separately from wrong answers
    items = [
        sql.score_item("SELECT a FROM t", "SELECT a FROM t"),        # correct
        sql.score_item("SELECT b FROM t", "SELECT a FROM t"),        # wrong
        sql.score_item("garbage ###", "SELECT a FROM t"),           # parse fail
    ]
    agg = sql.score_corpus(items)
    check("sql corpus accuracy 1/3", abs(agg["accuracy"] - 1 / 3) < 1e-9, str(agg))
    check("sql corpus parse_failure_rate 1/3",
          abs(agg["parse_failure_rate"] - 1 / 3) < 1e-9, str(agg))


# ---------------------------------------------------------------------------
def test_math():
    print("Math scorer")
    # gsm8k-style reference with '#### N'
    ref = "Natalia sold 48/2 = 24 clips in May.\n#### 72"
    r = math_scorer.score_item("She sold 48 + 24 = 72 clips.\nThe answer is 72.", ref)
    check("math correct final number", r["correct"] == 1, str(r))

    # formatting tolerance: $, commas, trailing period
    r = math_scorer.score_item("The total is $1,234.", "#### 1234")
    check("math $ + comma tolerated", r["correct"] == 1, str(r))

    # reasoning correct-looking but final number wrong
    r = math_scorer.score_item("2+2 = 5 so the answer is 5", "#### 4")
    check("math wrong final -> incorrect", r["correct"] == 0, str(r))

    # score the FINAL answer, not an intermediate number
    r = math_scorer.score_item("First 100, then subtract 58. The answer is 42.", "#### 42")
    check("math picks final not intermediate", r["correct"] == 1, str(r))

    # decimal equivalence
    r = math_scorer.score_item("answer: 42.0", "#### 42")
    check("math 42.0 == 42", r["correct"] == 1, str(r))

    # unparsable prediction distinguished
    r = math_scorer.score_item("I don't know", "#### 42")
    check("math no-number -> pred_unparsable",
          r["correct"] == 0 and r["reason"] == "pred_unparsable", str(r))


# ---------------------------------------------------------------------------
def test_translation():
    print("Translation scorer (chrF++)")
    perfect = [translation.score_item("The cat sat on the mat.", "The cat sat on the mat.")]
    agg_perfect = translation.score_corpus(perfect)
    check("chrf++ perfect ~ 100", agg_perfect["chrf++"] > 99.0, str(agg_perfect))

    bad = [translation.score_item("xyzzy plugh foobar", "The cat sat on the mat.")]
    agg_bad = translation.score_corpus(bad)
    check("chrf++ garbage low", agg_bad["chrf++"] < 20.0, str(agg_bad))

    # good > bad, and label stripping works
    good = [translation.score_item("Translation: The cat sat on the mat.",
                                   "The cat sat on the mat.")]
    agg_good = translation.score_corpus(good)
    check("chrf++ label stripped -> high", agg_good["chrf++"] > 99.0, str(agg_good))
    check("chrf++ good > bad", agg_good["chrf++"] > agg_bad["chrf++"])


# ---------------------------------------------------------------------------
def test_json():
    print("JSON scorer")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "in_stock": {"type": "boolean"},
        },
        "required": ["name", "price"],
    }
    gold = {"name": "Widget", "price": 9.99, "in_stock": True}

    # perfect
    r = json_extract.score_item('{"name": "Widget", "price": 9.99, "in_stock": true}',
                                "", {"schema": schema, "target": gold})
    check("json perfect valid", r["schema_valid"] and r["fp"] == 0 and r["fn"] == 0, str(r))

    # valid-but-empty must NOT score well: make fields optional so {} validates
    schema_opt = {"type": "object",
                  "properties": schema["properties"]}  # no 'required'
    r = json_extract.score_item("{}", "", {"schema": schema_opt, "target": gold})
    agg = json_extract.score_corpus([r])
    check("json empty validates but F1==0",
          r["schema_valid"] and agg["field_f1"] == 0.0, str((r, agg)))

    # schema-invalid: missing required field
    r = json_extract.score_item('{"in_stock": true}', "", {"schema": schema, "target": gold})
    check("json missing required -> invalid", not r["schema_valid"], str(r))

    # wrong value counts as fp+fn, not tp
    r = json_extract.score_item('{"name": "Gadget", "price": 9.99, "in_stock": true}',
                                "", {"schema": schema, "target": gold})
    check("json wrong value not counted correct", r["tp"] == 2 and r["fp"] == 1 and r["fn"] == 1,
          str(r))

    # tolerate fenced JSON + surrounding prose
    r = json_extract.score_item('Here you go:\n```json\n{"name":"Widget","price":9.99,"in_stock":true}\n```',
                                "", {"schema": schema, "target": gold})
    check("json fenced+prose parsed & perfect", r["schema_valid"] and r["fp"] == 0 and r["fn"] == 0,
          str(r))

    # non-JSON -> parse failure, all gold fields missed
    r = json_extract.score_item("no json here", "", {"schema": schema, "target": gold})
    check("json non-json -> parse fail, fn=3",
          (not r["parse_ok"]) and r["fn"] == 3, str(r))

    # corpus F1 sanity: one perfect + one empty over optional schema -> precision 1, recall 0.5
    items = [
        json_extract.score_item('{"name":"Widget","price":9.99,"in_stock":true}', "",
                                {"schema": schema_opt, "target": gold}),
        json_extract.score_item("{}", "", {"schema": schema_opt, "target": gold}),
    ]
    agg = json_extract.score_corpus(items)
    check("json corpus recall 0.5", abs(agg["field_recall"] - 0.5) < 1e-9, str(agg))
    check("json corpus precision 1.0", abs(agg["field_precision"] - 1.0) < 1e-9, str(agg))


def main() -> int:
    for fn in (test_sql, test_math, test_translation, test_json):
        fn()
    print()
    if FAILURES:
        print(f"SCORER VERIFICATION FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL SCORER CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
