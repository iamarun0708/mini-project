"""Build the merged-adapter grid with PEFT's weighted-adapter arithmetic.

    .venv/bin/python -m merge.build --dry-run          # list what would be built
    .venv/bin/python -m merge.build                    # all 55, skipping existing
    .venv/bin/python -m merge.build --methods ties     # one method (11 models)
    .venv/bin/python -m merge.build --arity 2          # pairs only
    .venv/bin/python -m merge.build --sweep --methods ties,linear

Cost note: merging never touches the 2.9 GB base weights numerically — it is
arithmetic on the rank-16 A/B factors, ~71 MB per adapter. The base model is
loaded on CPU only because PEFT needs the module structure to hang adapters on.
The whole 55-model grid is minutes of CPU, and leaves the GPU free.

Reproducibility: `dare_ties` and `dare_linear` drop delta entries at RANDOM.
Every merge therefore reseeds torch from a hash of (seed, method, combo,
weights) before calling into PEFT, so a given model id is byte-reproducible and
independent of the order the grid is built in. The other three methods are
deterministic and unaffected.

Caveat recorded in each merge_meta.json: PEFT's non-SVD `ties`/`dare_ties`
elect signs on the LoRA A and B factors SEPARATELY, not on the composed
delta (alpha/r)*B@A. That is standard PEFT behaviour but is not literally the
operation TIES defines for full checkpoints.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

from common import load_config, model_revision, set_all_seeds, sha256_text
from merge.grid import TASKS, enumerate_grid, enumerate_sweep

# Methods that take a `density` (fraction of entries retained). `linear` does not.
DENSITY_METHODS = {"ties", "dare_ties", "dare_linear", "magnitude_prune"}
# Temporary in-memory name for the freshly merged adapter.
MERGED_NAME = "merged"

# LoRA fields that MUST match across specialists for a merge to be meaningful.
# A mismatch here is a confound, not an inconvenience — fail loudly.
GEOMETRY_KEYS = ("r", "lora_alpha", "lora_dropout", "bias", "target_modules")


def merge_seed(base_seed: int, method: str, combo, weights) -> int:
    """Deterministic per-merge seed, stable across runs and build order."""
    key = f"{method}|{'-'.join(combo)}|{','.join(f'{w:.6f}' for w in weights)}"
    return (base_seed + int(sha256_text(key)[:8], 16)) % (2**31 - 1)


def adapter_geometry(adapter_dir: Path) -> dict:
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    geom = {k: cfg.get(k) for k in GEOMETRY_KEYS}
    if isinstance(geom["target_modules"], list):
        geom["target_modules"] = sorted(geom["target_modules"])
    return geom


def assert_uniform_geometry(adapters_dir: Path) -> dict:
    """Verify all four specialists share one LoRA geometry (paper §IV-C)."""
    geoms = {t: adapter_geometry(adapters_dir / t) for t in TASKS}
    ref_task, ref = next(iter(geoms.items()))
    for task, geom in geoms.items():
        if geom != ref:
            raise SystemExit(
                f"LoRA geometry mismatch: '{task}' differs from '{ref_task}'.\n"
                f"  {ref_task}: {ref}\n  {task}: {geom}\n"
                "Merging models with different adapter geometry would confound "
                "the comparison. Retrain with identical config.lora."
            )
    return ref


def load_stack(cfg: dict):
    """Load the base model on CPU and attach all four specialists."""
    base_dir = cfg["model"]["local_dir"]
    adapters_dir = Path(cfg["merge"]["adapters_dir"])

    print(f"[base ] {base_dir} (cpu, bfloat16)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(
        model, str(adapters_dir / TASKS[0]), adapter_name=TASKS[0]
    )
    for task in TASKS[1:]:
        model.load_adapter(str(adapters_dir / task), adapter_name=task)
    print(f"[stack] attached specialists: {', '.join(TASKS)}", flush=True)
    return model


def build_one(model, spec: dict, out_root: Path, cfg: dict, geometry: dict) -> float:
    """Merge one spec and write it to out_root/<model_id>/. Returns seconds."""
    method, combo, weights = spec["method"], list(spec["combo"]), spec["weights"]
    outdir = out_root / spec["model_id"]
    stage = out_root / f".{spec['model_id']}.partial"

    kwargs = {}
    if method in DENSITY_METHODS:
        kwargs["density"] = cfg["merge"]["density"]
    if method in {"ties", "dare_ties"}:
        kwargs["majority_sign_method"] = cfg["merge"]["majority_sign_method"]

    seed = merge_seed(cfg["seed"], method, combo, weights)
    torch.manual_seed(seed)

    t0 = time.time()
    if MERGED_NAME in model.peft_config:          # left over from a failed run
        model.delete_adapter(MERGED_NAME)
    model.add_weighted_adapter(
        adapters=combo,
        weights=weights,
        adapter_name=MERGED_NAME,
        combination_type=method,
        **kwargs,
    )
    try:
        # Stage then move, so an interrupted build never leaves a directory that
        # --resume would mistake for a finished one.
        if stage.exists():
            shutil.rmtree(stage)
        model.save_pretrained(str(stage), selected_adapters=[MERGED_NAME])

        meta = {
            "model_id": spec["model_id"],
            "method": method,
            "combo": combo,
            "weights": weights,
            "density": kwargs.get("density"),
            "majority_sign_method": kwargs.get("majority_sign_method"),
            "merge_seed": seed,
            "base_seed": cfg["seed"],
            "base_model": cfg["model"]["repo_id"],
            "base_revision": model_revision(cfg),
            "lora_geometry": geometry,
            "peft_version": __import__("peft").__version__,
            "torch_version": torch.__version__,
            "caveat": (
                "PEFT non-SVD ties/dare_ties elect signs on the LoRA A and B "
                "factors separately, not on the composed delta (alpha/r)*B@A."
            ),
        }
        (stage / MERGED_NAME / "merge_meta.json").write_text(json.dumps(meta, indent=2))

        if outdir.exists():
            shutil.rmtree(outdir)
        shutil.move(str(stage / MERGED_NAME), str(outdir))
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        model.delete_adapter(MERGED_NAME)

    return time.time() - t0


def main() -> int:
    cfg = load_config()
    mcfg = cfg["merge"]

    ap = argparse.ArgumentParser(description="Build the merged-adapter grid.")
    ap.add_argument("--methods", default=",".join(mcfg["peft_methods"]),
                    help="comma-separated subset of config.merge.peft_methods")
    ap.add_argument("--arity", type=int, choices=[2, 3, 4], default=None,
                    help="restrict to pairs (2), triples (3) or the quad (4)")
    ap.add_argument("--sweep", action="store_true",
                    help="build the coefficient sweep over pairs instead of the main grid")
    ap.add_argument("--out", default=mcfg["output_dir"])
    ap.add_argument("--limit", type=int, default=None, help="build at most N models")
    ap.add_argument("--force", action="store_true", help="rebuild models that already exist")
    ap.add_argument("--dry-run", action="store_true", help="list the grid and exit")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = set(methods) - set(mcfg["peft_methods"])
    if unknown:
        raise SystemExit(f"unknown method(s): {sorted(unknown)}; "
                         f"config allows {mcfg['peft_methods']}")

    if args.sweep:
        specs = enumerate_sweep(methods, mcfg["coefficient_sweep"]["weights"])
    else:
        specs = enumerate_grid(methods, arity=args.arity)

    out_root = Path(args.out)
    todo = [s for s in specs
            if args.force or not (out_root / s["model_id"] / "adapter_config.json").exists()]
    skipped = len(specs) - len(todo)
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"grid: {len(specs)} model(s)  |  to build: {len(todo)}  |  "
          f"already present: {skipped}")
    if args.dry_run:
        for s in specs:
            mark = "skip" if (out_root / s["model_id"] / "adapter_config.json").exists() else "build"
            print(f"  [{mark:5}] {s['model_id']:<44} weights={s['weights']}")
        return 0
    if not todo:
        print("nothing to do.")
        return 0

    set_all_seeds(cfg["seed"])
    geometry = assert_uniform_geometry(Path(mcfg["adapters_dir"]))
    print(f"[check] LoRA geometry uniform across specialists: r={geometry['r']}, "
          f"alpha={geometry['lora_alpha']}, {len(geometry['target_modules'])} modules")

    out_root.mkdir(parents=True, exist_ok=True)
    model = load_stack(cfg)

    t_start = time.time()
    for i, spec in enumerate(todo, 1):
        dt = build_one(model, spec, out_root, cfg, geometry)
        print(f"[{i:3}/{len(todo)}] {spec['model_id']:<44} {dt:5.1f}s", flush=True)

    total = time.time() - t_start
    print(f"\nbuilt {len(todo)} model(s) in {total/60:.1f} min -> {out_root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
