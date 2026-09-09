"""Secondary weight-space cross-check via mergekit (spec's "known blocker").

The main grid (`merge/build.py`) merges LoRA *adapters* with PEFT's
add_weighted_adapter -- cheap (rank-16 factors) but PEFT-specific machinery.
This script cross-checks a slice of that grid using mergekit's independent
implementation of task_arithmetic and ties, operating on full materialized
checkpoints instead of adapter factors. Same tasks, same equal weights, same
density (config.merge.density) -- different code path, so agreement is a
sanity check on the PEFT-grid numbers.

Two steps:
  1. Materialize each of the four LoRA specialists into a standalone full
     checkpoint (base + adapter, merged and unloaded). ~2.9 GB each.
  2. For each (method, combo) in the cross-check slice, write a mergekit YAML
     and run `mergekit-yaml` against the materialized checkpoints.

Cost: full fp/bf16 checkpoints, not rank-16 factors, so this is NOT the cheap
path -- ~3.1 GB per output, ~31 GB total for 10 configs. Runs on CPU by
default so it doesn't fight the GPU eval jobs for VRAM.

Run:
  .venv/bin/python -m merge.mergekit_crosscheck --materialize
  .venv/bin/python -m merge.mergekit_crosscheck --dry-run
  .venv/bin/python -m merge.mergekit_crosscheck
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, model_revision  # noqa: E402
from merge.grid import TASKS  # noqa: E402

# 5 combos (3 pairs + 1 triple + 1 quad) x 2 methods (task_arithmetic, ties)
# = 10 configs, matching the "~10 configs" budget noted in CLAUDE.md.
CROSSCHECK_COMBOS = [
    ("sql", "math"),
    ("sql", "translation"),
    ("math", "json"),
    ("sql", "math", "translation"),
    ("sql", "math", "translation", "json"),
]
CROSSCHECK_METHODS = ["task_arithmetic", "ties"]


def specialist_dir(root: Path, task: str) -> Path:
    return root / "_specialists" / task


def materialize_specialists(cfg: dict, root: Path, force: bool) -> None:
    """Merge each LoRA specialist onto the base and save a full checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_dir = cfg["model"]["local_dir"]
    adapters_dir = Path(cfg["merge"]["adapters_dir"])

    for task in TASKS:
        outdir = specialist_dir(root, task)
        if outdir.exists() and not force:
            print(f"[skip] {task} already materialized -> {outdir}")
            continue

        print(f"[materialize] {task}: base + adapter -> full checkpoint", flush=True)
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            base_dir, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(model, str(adapters_dir / task))
        model = model.merge_and_unload()

        outdir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(outdir), safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(base_dir)
        tokenizer.save_pretrained(str(outdir))
        del model
        print(f"  done in {time.time() - t0:.1f}s -> {outdir}")


def model_id(method: str, combo: tuple[str, ...]) -> str:
    return f"{method}__{'-'.join(combo)}"


def equal_weights(combo: tuple[str, ...]) -> list[float]:
    return [1.0 / len(combo)] * len(combo)


def write_yaml_config(cfg: dict, root: Path, method: str, combo: tuple[str, ...],
                       out_path: Path) -> Path:
    base_dir = str(Path(cfg["model"]["local_dir"]).resolve())
    weights = equal_weights(combo)
    density = cfg["merge"]["density"]

    models = []
    for task, w in zip(combo, weights):
        params = {"weight": w}
        if method == "ties":
            params["density"] = density
        models.append({"model": str(specialist_dir(root, task).resolve()),
                        "parameters": params})

    merge_cfg: dict = {
        "merge_method": method,
        "base_model": base_dir,
        "models": models,
        "dtype": "bfloat16",
    }
    if method == "ties":
        merge_cfg["parameters"] = {"normalize": True}

    out_path.write_text(yaml.safe_dump(merge_cfg, sort_keys=False))
    return out_path


def run_one(cfg: dict, root: Path, method: str, combo: tuple[str, ...],
            configs_dir: Path, out_root: Path, force: bool) -> float | None:
    mid = model_id(method, combo)
    outdir = out_root / mid
    if outdir.exists() and not force:
        print(f"[skip] {mid} already built")
        return None

    yaml_path = write_yaml_config(cfg, root, method, combo, configs_dir / f"{mid}.yaml")

    # mergekit 0.1.4 + this pydantic build leave ConfiguredModuleArchitecture's
    # forward refs unresolved unless `torch` is explicitly in scope when it is
    # rebuilt -- otherwise every run_yaml invocation dies with a
    # PydanticUserError ("you should define `torch`, then call model_rebuild()")
    # before it ever touches our config. Force the rebuild in-process first.
    wrapper = (
        "import sys, torch, pydantic\n"
        "import mergekit.architecture.base as _b\n"
        "for _name in ('ConfiguredModuleArchitecture', 'ConfiguredModelArchitecture'):\n"
        "    _obj = getattr(_b, _name, None)\n"
        "    if _obj is not None:\n"
        "        try:\n"
        "            _obj.model_rebuild(force=True, _types_namespace={'torch': torch})\n"
        "        except Exception as _e:\n"
        "            print('[rebuild-skip]', _name, repr(_e))\n"
        "from mergekit.scripts.run_yaml import main\n"
        f"sys.argv = ['mergekit-yaml', {str(yaml_path)!r}, {str(outdir)!r}, "
        "'--lora-merge-dtype', 'bfloat16']\n"
        "main()\n"
    )
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-c", wrapper],
        capture_output=True, text=True,
    )
    dt = time.time() - t0
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise SystemExit(f"mergekit-yaml failed for {mid} (exit {result.returncode})")

    meta = {
        "model_id": mid,
        "method": method,
        "combo": list(combo),
        "weights": equal_weights(combo),
        "density": cfg["merge"]["density"] if method == "ties" else None,
        "base_model": cfg["model"]["repo_id"],
        "base_revision": model_revision(cfg),
        "engine": "mergekit",
        "mergekit_version": __import__("mergekit").__version__
        if hasattr(__import__("mergekit"), "__version__") else "unknown",
        "note": (
            "Independent cross-check of the PEFT adapter-merge grid: this merges "
            "full materialized checkpoints via mergekit's task_arithmetic/ties, "
            "not LoRA A/B factors via PEFT add_weighted_adapter."
        ),
    }
    (outdir / "crosscheck_meta.json").write_text(json.dumps(meta, indent=2))
    return dt


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="mergekit cross-check of a slice of the merge grid.")
    ap.add_argument("--root", default="./mergekit_out",
                     help="root dir: holds _specialists/ (materialized checkpoints) and outputs")
    ap.add_argument("--materialize", action="store_true",
                     help="(re)materialize the 4 full-checkpoint specialists and exit")
    ap.add_argument("--force", action="store_true", help="rebuild existing outputs")
    ap.add_argument("--dry-run", action="store_true", help="list the plan and exit")
    args = ap.parse_args()

    root = Path(args.root)
    configs_dir = root / "_configs"
    out_root = root
    configs_dir.mkdir(parents=True, exist_ok=True)

    specs = [(m, c) for m in CROSSCHECK_METHODS for c in CROSSCHECK_COMBOS]
    print(f"cross-check plan: {len(specs)} config(s) "
          f"({len(CROSSCHECK_METHODS)} methods x {len(CROSSCHECK_COMBOS)} combos)")
    for m, c in specs:
        mark = "skip" if (out_root / model_id(m, c)).exists() and not args.force else "build"
        print(f"  [{mark:5}] {model_id(m, c)}")

    if args.dry_run:
        return 0

    materialize_specialists(cfg, root, force=args.materialize)
    if args.materialize:
        return 0

    for task in TASKS:
        if not (specialist_dir(root, task) / "config.json").exists():
            raise SystemExit(
                f"specialist '{task}' not materialized under {root}/_specialists/. "
                f"Run with --materialize first."
            )

    t_start = time.time()
    built = 0
    for method, combo in specs:
        dt = run_one(cfg, root, method, combo, configs_dir, out_root, args.force)
        if dt is not None:
            built += 1
            print(f"[{built:2}/{len(specs)}] {model_id(method, combo):<48} {dt/60:5.1f} min")

    print(f"\nbuilt {built} cross-check model(s) in {(time.time()-t_start)/60:.1f} min -> {out_root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
