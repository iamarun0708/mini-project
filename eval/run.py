"""Evaluation runner: score one (model, task) cell at a time.

    .venv/bin/python -m eval.run --models sql --tasks sql --limit 8   # calibrate
    .venv/bin/python -m eval.run --controls                           # 20 cells
    .venv/bin/python -m eval.run --shard 1/8                          # a slice
    .venv/bin/python -m eval.run --list                               # plan only

Design (see CLAUDE.md "Eval plan"):

* The atomic unit is a CELL = (model, task) = 300 greedy generations, ~5-10 min.
  60 models x 4 tasks = 240 cells. A crash costs at most one cell.
* PER-ITEM PREDICTIONS ARE PERSISTED and aggregation happens late. chrF++ is
  corpus-level and cannot be averaged across shards; JSON micro-F1 pools
  tp/fp/fn globally. Storing per-item records and calling score_corpus once at
  the end makes ANY sharding scheme exact.
* --resume skips finished cells, so shards can be run in any order, over days.
* Controls (base + 4 specialists) sort first: their scores are the DENOMINATOR
  of R_i = s_i(theta_M)/s_i(theta_i), so no retention number exists until they
  have run.

Greedy decoding throughout (config.eval.do_sample: false) — reproducibility over
peak scores, since the measured quantity is a difference between models.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_config, model_revision, set_all_seeds
from merge.grid import CONTROLS, TASKS, enumerate_grid
from train.prompt_templates import build_prompt
from eval.scorers import sql as sql_scorer
from eval.scorers import math as math_scorer
from eval.scorers import translation as translation_scorer
from eval.scorers import json_extract as json_scorer

SCORERS = {
    "sql": sql_scorer,
    "math": math_scorer,
    "translation": translation_scorer,
    "json": json_scorer,
}


# ---------------------------------------------------------------- model specs

def resolve_adapter(model_id: str, cfg: dict) -> Path | None:
    """Map a model id to its adapter dir. `base` means no adapter at all."""
    if model_id == "base":
        return None
    for root in (cfg["merge"]["adapters_dir"], cfg["merge"]["output_dir"]):
        cand = Path(root) / model_id
        if (cand / "adapter_config.json").exists():
            return cand
    raise SystemExit(
        f"no adapter found for model id '{model_id}' in "
        f"{cfg['merge']['adapters_dir']}/ or {cfg['merge']['output_dir']}/"
    )


def all_model_ids(cfg: dict) -> list[str]:
    """Controls first, then the merged grid — see module docstring."""
    merged = [s["model_id"] for s in enumerate_grid(cfg["merge"]["peft_methods"])]
    return list(CONTROLS) + merged


# ------------------------------------------------------------------ json glue

def extract_schema(input_text: str) -> dict | None:
    """Recover the JSON schema that prepare.py embedded in the prompt.

    The json scorer wants the schema in `extra` so it can run Draft7 validation,
    but the eval JSONL only carries the rendered prompt text, which looks like:

        Schema:
        {...}

        Text: ...

    Returns None if the block is absent or unparsable; the scorer then falls
    back to field matching without schema validation.
    """
    if "Schema:" not in input_text:
        return None
    after = input_text.split("Schema:", 1)[1].lstrip()
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(after):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(after[: i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def build_extra(task: str, row: dict) -> dict | None:
    if task != "json":
        return None
    extra = {"schema": extract_schema(row["input"])}
    try:
        extra["target"] = json.loads(row["output"])
    except json.JSONDecodeError:
        extra["target"] = None
    return extra


# ------------------------------------------------------------------ execution

def load_eval_rows(cfg: dict, task: str, limit: int | None) -> list[dict]:
    path = Path(cfg["data"]["processed_dir"]) / f"{task}_eval.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


@torch.inference_mode()
def generate(model, tokenizer, prompts: list[str], max_new_tokens: int,
             batch_size: int) -> list[str]:
    """Greedy batched generation. Returns only the newly generated text."""
    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        # Left padding => the prompt occupies a fixed prefix width for the batch.
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return outputs


def run_cell(model, tokenizer, cfg: dict, model_id: str, task: str,
             limit: int | None, results_dir: Path) -> dict:
    """Generate + score one (model, task) cell and write both output files."""
    rows = load_eval_rows(cfg, task, limit)
    instruction = cfg["tasks"][task]["instruction"]
    max_new = cfg["tasks"][task]["max_new_tokens"]

    prompts = [build_prompt(instruction, r["input"]) for r in rows]

    t0 = time.time()
    preds = generate(model, tokenizer, prompts, max_new, cfg["eval"]["batch_size"])
    gen_secs = time.time() - t0

    scorer = SCORERS[task]
    per_item = [
        scorer.score_item(pred, row["output"], build_extra(task, row))
        for pred, row in zip(preds, rows)
    ]
    metrics = scorer.score_corpus(per_item)

    outdir = results_dir / model_id
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / f"{task}.preds.jsonl").open("w") as fh:
        for row, pred, item in zip(rows, preds, per_item):
            fh.write(json.dumps({"input": row["input"], "reference": row["output"],
                                 "prediction": pred, "item": item}) + "\n")

    record = {
        "model_id": model_id,
        "task": task,
        "metrics": metrics,
        "n_items": len(rows),
        "limited": limit is not None,
        "gen_seconds": round(gen_secs, 1),
        "secs_per_item": round(gen_secs / max(len(rows), 1), 2),
        "max_new_tokens": max_new,
        "batch_size": cfg["eval"]["batch_size"],
        "do_sample": cfg["eval"]["do_sample"],
        "base_revision": model_revision(cfg),
        "seed": cfg["seed"],
    }
    (outdir / f"{task}.json").write_text(json.dumps(record, indent=2))
    return record


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Run evaluation cells.")
    ap.add_argument("--models", default=None,
                    help="comma-separated model ids (default: controls + full grid)")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--controls", action="store_true",
                    help="shorthand for --models base,sql,math,translation,json")
    ap.add_argument("--shard", default=None, metavar="K/N",
                    help="run only shard K of N over the cell list (1-indexed)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap items per cell — use to calibrate timing")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="re-run finished cells")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    if args.batch_size:
        cfg["eval"]["batch_size"] = args.batch_size
    results_dir = Path(args.out or cfg["eval"]["results_dir"])

    if args.controls:
        model_ids = list(CONTROLS)
    elif args.models:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_ids = all_model_ids(cfg)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if set(tasks) - set(TASKS):
        raise SystemExit(f"unknown task(s): {sorted(set(tasks) - set(TASKS))}")

    cells = [(m, t) for m in model_ids for t in tasks]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        if not 1 <= k <= n:
            raise SystemExit(f"--shard K/N requires 1 <= K <= N, got {args.shard}")
        lo = (k - 1) * len(cells) // n
        hi = k * len(cells) // n
        cells = cells[lo:hi]
        print(f"shard {k}/{n}: cells [{lo}:{hi}]")

    todo = [c for c in cells
            if args.force or not (results_dir / c[0] / f"{c[1]}.json").exists()]
    print(f"cells: {len(cells)} selected  |  to run: {len(todo)}  |  "
          f"done: {len(cells) - len(todo)}")
    if args.list:
        for m, t in cells:
            mark = "done" if (results_dir / m / f"{t}.json").exists() else "run"
            print(f"  [{mark:4}] {m} / {t}")
        return 0
    if not todo:
        print("nothing to do.")
        return 0

    set_all_seeds(cfg["seed"])
    base_dir = cfg["model"]["local_dir"]
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    # Decoder-only batched generation requires LEFT padding, else the generated
    # continuation starts after pad tokens and the prompt slice is misaligned.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[base ] {base_dir} (cuda, bfloat16)", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_dir, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True
    )
    base.eval()

    # Group by model so each adapter is attached once, not once per task.
    by_model: dict[str, list[str]] = {}
    for m, t in todo:
        by_model.setdefault(m, []).append(t)

    t_start = time.time()
    done = 0
    for model_id, model_tasks in by_model.items():
        adapter = resolve_adapter(model_id, cfg)
        if adapter is None:
            model = base
        else:
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, str(adapter))
            model.eval()
        print(f"\n=== {model_id} ({'no adapter' if adapter is None else adapter}) ===",
              flush=True)

        for task in model_tasks:
            rec = run_cell(model, tokenizer, cfg, model_id, task,
                           args.limit, results_dir)
            done += 1
            headline = {k: v for k, v in rec["metrics"].items() if k != "n"}
            print(f"[{done:3}/{len(todo)}] {task:<12} "
                  f"{rec['gen_seconds']:6.1f}s  ({rec['secs_per_item']:.2f}s/item)  "
                  f"{headline}", flush=True)

        if adapter is not None:
            model.unload()
            del model
            torch.cuda.empty_cache()

    print(f"\nran {done} cell(s) in {(time.time() - t_start)/60:.1f} min "
          f"-> {results_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
