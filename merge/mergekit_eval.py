"""Evaluate the mergekit cross-check models and compare against the PEFT grid.

`merge/mergekit_crosscheck.py` only builds the 10 cross-check checkpoints; it
does not evaluate them, so on its own it proves nothing. This script closes
the loop: run the same greedy-generation eval used everywhere else in the
project (eval/scorers, eval/run.generate, train/prompt_templates.build_prompt)
against each mergekit checkpoint, then compare retention against the PEFT
adapter-merge grid's equivalent (method, combo).

mergekit's `ties` maps directly to PEFT's `ties`. mergekit's `task_arithmetic`
has no exact PEFT equivalent in config.merge.peft_methods; it is compared
against PEFT's `linear` as the closest conceptual match (both are an
unweighted/equal-weighted combination without TIES-style sign election or
sparsification -- task_arithmetic is literally "linear in delta space").

Only the tasks in each model's own combo are evaluated (not all 4), since
this is a secondary sanity check, not a full second grid -- see CLAUDE.md
"known blockers": mergekit merges full checkpoints (2.9 GB+ each), so it is
not the cheap path the PEFT grid is.

Run:  .venv/bin/python -m merge.mergekit_eval
      .venv/bin/python -m merge.mergekit_eval --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, set_all_seeds  # noqa: E402
from train.prompt_templates import build_prompt  # noqa: E402
from eval.run import generate, build_extra, load_eval_rows  # noqa: E402
from eval.scorers import sql as sql_scorer  # noqa: E402
from eval.scorers import math as math_scorer  # noqa: E402
from eval.scorers import translation as translation_scorer  # noqa: E402
from eval.scorers import json_extract as json_scorer  # noqa: E402

SCORERS = {
    "sql": sql_scorer, "math": math_scorer,
    "translation": translation_scorer, "json": json_scorer,
}
PRIMARY_METRIC = {"sql": "accuracy", "math": "accuracy",
                  "translation": "chrf++", "json": "field_f1"}

# method name in mergekit -> nearest equivalent in the PEFT grid
PEFT_EQUIVALENT = {"ties": "ties", "task_arithmetic": "linear"}


def crosscheck_specs(root: Path) -> list[dict]:
    specs = []
    for d in sorted(root.iterdir()):
        meta_path = d / "crosscheck_meta.json"
        if meta_path.exists():
            specs.append(json.loads(meta_path.read_text()))
    return specs


def run_cell(cfg: dict, model, tokenizer, model_id: str, task: str,
             results_dir: Path) -> dict:
    rows = load_eval_rows(cfg, task, None)
    instruction = cfg["tasks"][task]["instruction"]
    max_new = cfg["tasks"][task]["max_new_tokens"]
    prompts = [build_prompt(instruction, r["input"]) for r in rows]

    t0 = time.time()
    preds = generate(model, tokenizer, prompts, max_new, cfg["eval"]["batch_size"])
    gen_secs = time.time() - t0

    scorer = SCORERS[task]
    per_item = [scorer.score_item(pred, row["output"], build_extra(task, row))
                for pred, row in zip(preds, rows)]
    metrics = scorer.score_corpus(per_item)

    outdir = results_dir / model_id
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / f"{task}.preds.jsonl").open("w") as fh:
        for row, pred, item in zip(rows, preds, per_item):
            fh.write(json.dumps({"input": row["input"], "reference": row["output"],
                                 "prediction": pred, "item": item}) + "\n")
    record = {"model_id": model_id, "task": task, "metrics": metrics,
              "n_items": len(rows), "gen_seconds": round(gen_secs, 1), "engine": "mergekit"}
    (outdir / f"{task}.json").write_text(json.dumps(record, indent=2))
    return record


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./mergekit_out")
    ap.add_argument("--results-out", default="./results_mergekit")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    results_dir = Path(args.results_out)
    specs = crosscheck_specs(root)

    cells = [(s["model_id"], t) for s in specs for t in s["combo"]]
    todo = [(m, t) for m, t in cells
            if args.force or not (results_dir / m / f"{t}.json").exists()]
    print(f"cells: {len(cells)}  |  to run: {len(todo)}  |  done: {len(cells)-len(todo)}")
    if args.dry_run:
        for m, t in cells:
            mark = "done" if (results_dir / m / f"{t}.json").exists() else "run"
            print(f"  [{mark:4}] {m} / {t}")
        return 0
    if not todo:
        print("nothing to do.")
    else:
        set_all_seeds(cfg["seed"])
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["local_dir"])
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        by_model: dict[str, list[str]] = {}
        for m, t in todo:
            by_model.setdefault(m, []).append(t)

        done = 0
        for model_id, tasks in by_model.items():
            print(f"\n=== {model_id} ===", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(root / model_id), dtype=torch.bfloat16, device_map="cuda",
            )
            model.eval()
            for task in tasks:
                rec = run_cell(cfg, model, tokenizer, model_id, task, results_dir)
                done += 1
                print(f"[{done:2}/{len(todo)}] {task:<12} {rec['gen_seconds']:6.1f}s  "
                      f"{ {k:v for k,v in rec['metrics'].items() if k!='n'} }")
            del model
            torch.cuda.empty_cache()

    # ---- comparison against the PEFT grid
    peft_results = Path(cfg["eval"]["results_dir"])
    specialist_score = {
        t: json.loads((peft_results / t / f"{t}.json").read_text())["metrics"][PRIMARY_METRIC[t]]
        for t in ["sql", "math", "translation", "json"]
    }

    comparison = []
    for spec in specs:
        mid, method, combo = spec["model_id"], spec["method"], spec["combo"]
        peft_method = PEFT_EQUIVALENT[method]
        peft_model_id = f"{peft_method}__{'-'.join(combo)}"
        for task in combo:
            mk_path = results_dir / mid / f"{task}.json"
            peft_path = peft_results / peft_model_id / f"{task}.json"
            if not (mk_path.exists() and peft_path.exists()):
                continue
            mk_score = json.loads(mk_path.read_text())["metrics"][PRIMARY_METRIC[task]]
            peft_score = json.loads(peft_path.read_text())["metrics"][PRIMARY_METRIC[task]]
            denom = specialist_score[task]
            comparison.append({
                "mergekit_method": method, "peft_method": peft_method,
                "combo": "-".join(combo), "task": task,
                "mergekit_score": mk_score, "peft_score": peft_score,
                "mergekit_retention": mk_score / denom if denom else float("nan"),
                "peft_retention": peft_score / denom if denom else float("nan"),
            })

    out_dir = Path("analysis")
    out_dir.mkdir(exist_ok=True)
    if comparison:
        with (out_dir / "mergekit_vs_peft.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(comparison[0].keys()))
            w.writeheader()
            w.writerows(comparison)
        print(f"\n[write] {out_dir/'mergekit_vs_peft.csv'} ({len(comparison)} rows)")

        diffs = [abs(r["mergekit_retention"] - r["peft_retention"]) for r in comparison]
        mk_vals = [r["mergekit_retention"] for r in comparison]
        peft_vals = [r["peft_retention"] for r in comparison]
        n = len(mk_vals)
        mean_mk, mean_peft = sum(mk_vals) / n, sum(peft_vals) / n
        cov = sum((a - mean_mk) * (b - mean_peft) for a, b in zip(mk_vals, peft_vals)) / n
        std_mk = (sum((a - mean_mk) ** 2 for a in mk_vals) / n) ** 0.5
        std_peft = (sum((b - mean_peft) ** 2 for b in peft_vals) / n) ** 0.5
        corr = cov / (std_mk * std_peft) if std_mk and std_peft else float("nan")

        print(f"n={n}  mean|Δretention|={sum(diffs)/n:.4f}  max|Δretention|={max(diffs):.4f}  "
              f"pearson_r={corr:.4f}")
        print("(high correlation + small |delta| = mergekit's independent implementation "
              "agrees with the PEFT adapter-merge grid; this is the point of the cross-check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
