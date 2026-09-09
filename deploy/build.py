"""GGUF deployment (spec §11, stage 8).

Picks the best-performing merged model from analysis/retention.csv (highest
mean retention among the quad -- sql+math+translation+json -- merges, since
that is the only slice of the grid that actually covers all four tasks and is
therefore the only one meaningful to ship as a single general-purpose model),
materializes it as a full checkpoint (base + merged LoRA, merged and
unloaded), converts to GGUF f16 via llama.cpp, and quantizes to the format
pinned in config.yaml (deploy.quantization, default Q4_K_M).

Requires `llama.cpp/` cloned next to this repo with `llama-quantize` and
`convert_hf_to_gguf.py` available (see CLAUDE.md "known blockers"), and the
`gguf` python package (`pip install gguf` -- deliberately NOT the full
llama.cpp requirements*.txt, which pins an incompatible torch/transformers).

Run:  .venv/bin/python -m deploy.build
      .venv/bin/python -m deploy.build --model-id linear__sql-math   # override
      .venv/bin/python -m deploy.build --quantization Q8_0
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LLAMA_CPP_DIR = REPO_ROOT / "llama.cpp"


def find_quantize_binary() -> Path:
    for candidate in [
        LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-quantize.exe",
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize",
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize.exe",
    ]:
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"llama-quantize not found under {LLAMA_CPP_DIR}/build/. "
        f"Build it first (see CLAUDE.md known blockers): "
        f"cmake -B build -G \"Visual Studio 17 2022\" -A x64 -DGGML_CUDA=OFF "
        f"&& cmake --build build --config Release --target llama-quantize llama-cli"
    )


def best_quad_model(analysis_dir: Path) -> str:
    """The quad merge (all 4 tasks) with the highest mean retention.

    Only quad combos are considered: a pair or triple merge is not a general-
    purpose deployable model (it never learned some of the tasks), so ranking
    across arities and picking a pair would ship a model with zero SQL ability,
    for instance. Ties/pairs/triples remain useful for the *analysis*, just not
    as this pipeline's single deployment artifact.
    """
    retention_path = analysis_dir / "retention.csv"
    if not retention_path.exists():
        raise SystemExit(
            f"{retention_path} not found. Run: .venv/bin/python -m analysis.report"
        )
    rows = list(csv.DictReader(retention_path.open()))
    quad_rows = [r for r in rows if r["arity"] == "4"]
    if not quad_rows:
        raise SystemExit("no quad (4-task) merged models found in analysis/retention.csv")

    by_model: dict[str, list[float]] = {}
    for r in quad_rows:
        by_model.setdefault(r["model_id"], []).append(float(r["retention"]))
    ranked = sorted(by_model.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
    best_id, vals = ranked[0]
    print(f"[select] {best_id}  mean_retention={sum(vals)/len(vals):.4f}  "
          f"(best of {len(ranked)} quad merges)")
    return best_id


def materialize(cfg: dict, model_id: str, out_dir: Path, force: bool) -> Path:
    outdir = out_dir / "_materialized" / model_id
    if outdir.exists() and not force:
        print(f"[skip] already materialized -> {outdir}")
        return outdir

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_dir = cfg["model"]["local_dir"]
    adapter_dir = Path(cfg["merge"]["output_dir"]) / model_id
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"no merged adapter at {adapter_dir}; run merge.build first")

    print(f"[materialize] {model_id}: base + merged adapter -> full checkpoint", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()

    outdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(outdir), safe_serialization=True)
    AutoTokenizer.from_pretrained(base_dir).save_pretrained(str(outdir))
    del model
    print(f"  -> {outdir}")
    return outdir


def convert_to_gguf(materialized_dir: Path, out_path: Path, force: bool) -> Path:
    if out_path.exists() and not force:
        print(f"[skip] f16 gguf already exists -> {out_path}")
        return out_path
    script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not script.exists():
        raise SystemExit(f"{script} not found -- clone llama.cpp next to this repo")
    result = subprocess.run(
        [sys.executable, str(script), str(materialized_dir),
         "--outfile", str(out_path), "--outtype", "f16"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise SystemExit(f"convert_hf_to_gguf.py failed (exit {result.returncode})")
    print(f"[convert] -> {out_path}")
    return out_path


def quantize(f16_path: Path, out_path: Path, quantization: str, force: bool) -> Path:
    if out_path.exists() and not force:
        print(f"[skip] quantized gguf already exists -> {out_path}")
        return out_path
    quantize_bin = find_quantize_binary()
    result = subprocess.run(
        [str(quantize_bin), str(f16_path), str(out_path), quantization],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise SystemExit(f"llama-quantize failed (exit {result.returncode})")
    print(f"[quantize] {quantization} -> {out_path}")
    return out_path


def main() -> int:
    cfg = load_config()
    dcfg = cfg["deploy"]

    ap = argparse.ArgumentParser(description="Materialize, convert, and quantize a merged model to GGUF.")
    ap.add_argument("--model-id", default=None,
                     help="merged model id to deploy (default: best quad merge by mean retention)")
    ap.add_argument("--quantization", default=dcfg["quantization"])
    ap.add_argument("--out", default=dcfg["gguf_dir"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = args.model_id or best_quad_model(REPO_ROOT / "analysis")

    materialized = materialize(cfg, model_id, out_dir, args.force)
    f16_path = out_dir / f"{model_id}-f16.gguf"
    convert_to_gguf(materialized, f16_path, args.force)
    quant_path = out_dir / f"{model_id}-{args.quantization}.gguf"
    quantize(f16_path, quant_path, args.quantization, args.force)

    size_mb = quant_path.stat().st_size / 1e6
    print(f"\ndeployed: {quant_path} ({size_mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
