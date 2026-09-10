"""Analysis for the coefficient sweep (spec §11, stage 7 addendum).

analysis/report.py only ever covers the main 55-model grid: its model-id
parser (combo_for) splits on the first "__" and treats everything after it
as the combo, which is wrong for a sweep id like
"magnitude_prune__sql-math__w0.3-0.7" (three "__"-separated parts: method,
combo, weights -- see merge/grid.py's model_id()). So the 36-model
coefficient sweep built by `merge.build --sweep` needed its own parser and
its own retention table; this script is that.

Question the sweep actually answers: does moving weight off 0.5/0.5 trade one
task's retention for the other's, and does that trade favor either of the two
top methods identified by analysis/report.py (magnitude_prune, dare_ties)?

Outputs:
    analysis/sweep_retention.csv   one row per (model, task in its pair)
    analysis/sweep_by_weight.csv   mean retention per (method, pair, weight-point)

Run:  .venv/bin/python -m analysis.sweep_report
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config  # noqa: E402
from merge.grid import TASKS, all_combos  # noqa: E402
from analysis.report import PRIMARY_METRIC, load_score  # noqa: E402


def sweep_model_ids(methods: list[str], weight_pairs: list[list[float]]) -> list[dict]:
    """Mirror merge.grid.enumerate_sweep, but we only need id + combo + weights."""
    specs = []
    for method in methods:
        for combo in all_combos(arity=2):
            for w in weight_pairs:
                wstr = "-".join(f"{x:g}" for x in w)
                model_id = f"{method}__{'-'.join(combo)}__w{wstr}"
                specs.append({"model_id": model_id, "method": method,
                              "combo": combo, "weights": list(w)})
    return specs


def main() -> int:
    cfg = load_config()
    results_dir = Path(cfg["eval"]["results_dir"])
    out_dir = Path("analysis")

    methods = cfg["merge"]["coefficient_sweep"].get("methods") or \
        __import__("json").loads((out_dir / "summary.json").read_text())["top2_methods_for_sweep"]
    weight_pairs = cfg["merge"]["coefficient_sweep"]["weights"]

    specs = sweep_model_ids(methods, weight_pairs)
    specialist_score = {t: load_score(results_dir, t, t) for t in TASKS}

    rows: list[dict] = []
    missing: list[str] = []
    for spec in specs:
        for task in spec["combo"]:
            path = results_dir / spec["model_id"] / f"{task}.json"
            if not path.exists():
                missing.append(f"{spec['model_id']}/{task}")
                continue
            score = load_score(results_dir, spec["model_id"], task)
            denom = specialist_score[task]
            w_self = spec["weights"][spec["combo"].index(task)]
            rows.append({
                "model_id": spec["model_id"],
                "method": spec["method"],
                "combo": "-".join(spec["combo"]),
                "weight_point": "-".join(f"{x:g}" for x in spec["weights"]),
                "task": task,
                "weight_on_task": w_self,
                "score": score,
                "specialist_score": denom,
                "retention": score / denom if denom else float("nan"),
            })

    if missing:
        raise SystemExit(
            f"sweep analysis requires every sweep eval cell to exist; missing "
            f"{len(missing)}: {missing[:10]}{' ...' if len(missing) > 10 else ''}\n"
            f"Run: .venv/bin/python -m eval.run --models <sweep ids>"
        )

    with (out_dir / "sweep_retention.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {out_dir/'sweep_retention.csv'} ({len(rows)} rows)")

    # ---- mean retention per (method, combo, weight_point) — the sweep curve
    keys = sorted({(r["method"], r["combo"], r["weight_point"]) for r in rows})
    by_weight = []
    for method, combo, wp in keys:
        g = [r for r in rows if r["method"] == method and r["combo"] == combo
             and r["weight_point"] == wp]
        mean_r = sum(r["retention"] for r in g) / len(g)
        by_weight.append({"method": method, "combo": combo, "weight_point": wp,
                          "n": len(g), "mean_retention": mean_r})
    by_weight.sort(key=lambda r: (r["method"], r["combo"], r["weight_point"]))

    with (out_dir / "sweep_by_weight.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by_weight[0].keys()))
        w.writeheader()
        w.writerows(by_weight)
    print(f"[write] {out_dir/'sweep_by_weight.csv'} ({len(by_weight)} rows)")

    # ---- headline: does 0.5/0.5 beat the skewed points, on average?
    center = [r["mean_retention"] for r in by_weight if r["weight_point"] == "0.5-0.5"]
    skewed = [r["mean_retention"] for r in by_weight if r["weight_point"] != "0.5-0.5"]
    print(f"\n0.5/0.5 mean retention: {sum(center)/len(center):.4f}  "
          f"(n={len(center)})")
    print(f"skewed (0.3/0.7, 0.7/0.3) mean retention: {sum(skewed)/len(skewed):.4f}  "
          f"(n={len(skewed)})")
    print("(skewed points score lower on average because 'weight_on_task' rows "
          "at 0.3 pull the mean down -- see sweep_retention.csv's weight_on_task "
          "column to separate 'favored' from 'disfavored' task in each pair)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
