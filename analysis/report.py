"""Analysis for the merge study (spec §11, stage 7).

Reads every (model, task) cell written by eval/run.py and computes the
retention ratio (paper eq. 1):

    R_i(M) = s_i(theta_M) / s_i(theta_i)

where theta_i is the single-task specialist for task i and theta_M is a merged
model. R_i(M) is only defined for tasks i that are actually part of M's combo
(merging sql+math tells you nothing about M's translation ability relative to
a translation specialist it never absorbed) -- see `combo_for`.

Outputs (all under ./analysis/, gitignored is NOT set for this dir so results
are committed):

    analysis/retention.csv        one row per (model, task in its combo)
    analysis/by_method.csv        mean/median retention per merge method
    analysis/by_arity.csv         mean retention per method x combo arity
    analysis/summary.json         headline numbers, incl. the top-2 methods
                                   that feed merge/build.py --sweep
    analysis/plots/retention_by_method.png
    analysis/plots/retention_by_arity.png

Run:  .venv/bin/python -m analysis.report
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config  # noqa: E402
from merge.grid import TASKS, CONTROLS, enumerate_grid  # noqa: E402

PRIMARY_METRIC = {
    "sql": "accuracy",
    "math": "accuracy",
    "translation": "chrf++",
    "json": "field_f1",
}


def combo_for(model_id: str) -> tuple[str, ...] | None:
    """Which tasks does this model id actually carry an adapter for?

    Controls: 'base' -> () (no specialist knowledge at all, excluded from
    retention -- there is no theta_i identity to compare it against beyond
    the specialists themselves). Individual specialists -> (task,). Merged
    grid ids -> the combo baked into merge/grid.py's model_id().
    """
    if model_id == "base":
        return ()
    if model_id in TASKS:
        return (model_id,)
    # merged id: "<method>__<task1>-<task2>[-...]"
    if "__" not in model_id:
        raise ValueError(f"unrecognized model id: {model_id}")
    _, combo_str = model_id.split("__", 1)
    return tuple(combo_str.split("-"))


def method_for(model_id: str) -> str | None:
    if model_id in CONTROLS:
        return None
    return model_id.split("__", 1)[0]


def load_score(results_dir: Path, model_id: str, task: str) -> float:
    path = results_dir / model_id / f"{task}.json"
    rec = json.loads(path.read_text())
    return rec["metrics"][PRIMARY_METRIC[task]]


def main() -> int:
    cfg = load_config()
    results_dir = Path(cfg["eval"]["results_dir"])
    out_dir = Path("analysis")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    merged_ids = [s["model_id"] for s in enumerate_grid(cfg["merge"]["peft_methods"])]
    all_ids = list(CONTROLS) + merged_ids

    # Specialist scores are the retention denominator (eq. 1's theta_i).
    specialist_score = {
        task: load_score(results_dir, task, task) for task in TASKS
    }

    rows: list[dict] = []
    missing: list[str] = []
    for model_id in all_ids:
        combo = combo_for(model_id)
        method = method_for(model_id)
        for task in combo:
            path = results_dir / model_id / f"{task}.json"
            if not path.exists():
                missing.append(f"{model_id}/{task}")
                continue
            score = load_score(results_dir, model_id, task)
            denom = specialist_score[task]
            retention = score / denom if denom != 0 else float("nan")
            rows.append({
                "model_id": model_id,
                "method": method,        # None for the 4 specialist controls
                "combo": "-".join(combo),
                "arity": len(combo),
                "task": task,
                "score": score,
                "specialist_score": denom,
                "retention": retention,
            })

    if missing:
        raise SystemExit(
            f"analysis requires every eval cell to exist; missing {len(missing)}: "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}\n"
            f"Run: .venv/bin/python -m eval.run"
        )

    out_dir.mkdir(exist_ok=True)
    with (out_dir / "retention.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {out_dir/'retention.csv'} ({len(rows)} rows)")

    # ---- aggregate: mean/median retention per method (specialists excluded,
    # they are the R=1.0 reference points by construction, not merge products)
    merged_rows = [r for r in rows if r["method"] is not None]

    def agg(group_rows: list[dict]) -> dict:
        vals = [r["retention"] for r in group_rows]
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        median = (vals_sorted[n // 2] if n % 2 else
                  (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2)
        return {
            "n": n,
            "mean_retention": sum(vals) / n,
            "median_retention": median,
            "min_retention": min(vals),
            "max_retention": max(vals),
        }

    methods = sorted({r["method"] for r in merged_rows})
    by_method = []
    for m in methods:
        g = [r for r in merged_rows if r["method"] == m]
        by_method.append({"method": m, **agg(g)})
    by_method.sort(key=lambda r: -r["mean_retention"])

    with (out_dir / "by_method.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by_method[0].keys()))
        w.writeheader()
        w.writerows(by_method)
    print(f"[write] {out_dir/'by_method.csv'}")
    for r in by_method:
        print(f"  {r['method']:<16} mean={r['mean_retention']:.4f}  "
              f"median={r['median_retention']:.4f}  n={r['n']}")

    arities = sorted({r["arity"] for r in merged_rows})
    by_arity = []
    for m in methods:
        for a in arities:
            g = [r for r in merged_rows if r["method"] == m and r["arity"] == a]
            if not g:
                continue
            by_arity.append({"method": m, "arity": a, **agg(g)})

    with (out_dir / "by_arity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by_arity[0].keys()))
        w.writeheader()
        w.writerows(by_arity)
    print(f"[write] {out_dir/'by_arity.csv'}")

    # ---- per-task breakdown too (interference is task-specific)
    tasks_present = sorted({r["task"] for r in merged_rows})
    by_method_task = []
    for m in methods:
        for t in tasks_present:
            g = [r for r in merged_rows if r["method"] == m and r["task"] == t]
            if not g:
                continue
            by_method_task.append({"method": m, "task": t, **agg(g)})
    with (out_dir / "by_method_task.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by_method_task[0].keys()))
        w.writeheader()
        w.writerows(by_method_task)
    print(f"[write] {out_dir/'by_method_task.csv'}")

    top2 = [r["method"] for r in by_method[:2]]
    summary = {
        "primary_metric": PRIMARY_METRIC,
        "specialist_scores": specialist_score,
        "base_scores": {
            task: load_score(results_dir, "base", task) for task in TASKS
        },
        "n_cells_analyzed": len(rows),
        "methods_ranked_by_mean_retention": [r["method"] for r in by_method],
        "top2_methods_for_sweep": top2,
        "note": (
            "R_i(M) = s_i(theta_M)/s_i(theta_i) is computed only for tasks i "
            "actually in M's merge combo. 'base' is reported for reference but "
            "excluded from method aggregates (it is not a merge product)."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[write] {out_dir/'summary.json'}")
    print(f"\ntop-2 methods for coefficient sweep: {top2}")
    print(f"  -> .venv/bin/python -m merge.build --sweep --methods {','.join(top2)}")

    # ---- plots (best-effort; analysis CSVs above are the load-bearing output)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([r["method"] for r in by_method],
               [r["mean_retention"] for r in by_method],
               yerr=None, color="#4C72B0")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1,
                    label="R=1.0 (full specialist retention)")
        ax.set_ylabel("mean retention R_i(M)")
        ax.set_title("Mean task retention by merge method (all combos, all tasks)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "retention_by_method.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for m in methods:
            xs = [r["arity"] for r in by_arity if r["method"] == m]
            ys = [r["mean_retention"] for r in by_arity if r["method"] == m]
            ax.plot(xs, ys, marker="o", label=m)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("combo arity (# tasks merged)")
        ax.set_ylabel("mean retention R_i(M)")
        ax.set_xticks(sorted(arities))
        ax.set_title("Retention vs. combo arity, by method")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "retention_by_arity.png", dpi=150)
        plt.close(fig)
        print(f"[write] {plots_dir}/retention_by_method.png, retention_by_arity.png")
    except ImportError:
        print("[skip] matplotlib not available, skipping plots")

    return 0


if __name__ == "__main__":
    sys.exit(main())
