# CLAUDE.md — read this first

Onboarding + live status for this project. Skim this before crawling the tree.
`.claudeignore` keeps the heavy dirs (`.venv/` ~5.9G, `models/` ~2.9G, generated
artifacts) out of the way.

## What this project is

**Systematic study of model merging in small language models.** Train four
task-specialist LoRA adapters on ONE pinned base model, merge them with several
weight-space methods, and measure task interference vs. transfer. Everything is
built as a controlled experiment: identical base snapshot, identical LoRA shapes,
identical prompt template, equal data budget per task — so any difference in a
merged model is attributable to the merge, not a confound.

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`, pinned to commit
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (config.model.revision). Already
  downloaded to `models/qwen2.5-1.5b-instruct/`.
- **Four tasks:** `sql`, `math`, `translation` (Tamil→English), `json` (extraction).
- **Hardware:** RTX 4070 Laptop, **8 GB VRAM** (spec assumed 12 GB). Adapt by
  lowering batch / raising grad-accum — never by switching models. Effective batch
  held at 32.

## Single source of truth

`config.yaml` — all paths, seeds, hyperparameters, task instructions, merge grid,
the pinned model revision. Read it before changing anything. `common.py` loads it
and applies global seeding (seed 42) + train/eval overlap hashing.

## Conventions

- **venv:** run everything with `.venv/bin/python`. Never `pip install` outside it.
- **"scripts" are flat root files** with a `scripts_` prefix emulating a `scripts/`
  dir: `scripts_download_model.py`, `scripts_install.sh`. Real packages are dirs:
  `eval/`, `train/`.
- **Prompt template is the load-bearing control** (`train/prompt_templates.py`):
  identical structure for all adapters + all eval; only the per-task `instruction`
  string differs. Loss is masked to the assistant turn only.
- **Eval is greedy** (`do_sample: false`) everywhere — reproducibility over peak scores.

## Pipeline status  (update this as stages complete)

| Stage | Location | Status |
|-------|----------|--------|
| 0. Install stack | `scripts_install.sh` | ✅ done — torch 2.6+cu124, transformers, peft 0.19, trl 1.9, datasets 5.0, mergekit, sqlglot, sacrebleu, jsonschema |
| 1. Download + pin base model | `scripts_download_model.py` | ✅ done — model present, revision pinned |
| 2. Scorers | `eval/scorers/{sql,math,translation,json_extract}.py` | ✅ done — gate `python -m eval.test_scorers` passes 24/24 |
| 3. Data preparation | `data/prepare.py` | ✅ done — all 4 datasets in `data/processed/` |
| 4. Train LoRA adapters | `train/run.py` → `adapters/` | ✅ 4/4 — sql 0.0186, math 0.2713, translation 1.8395, json 7.53e-06 (all @ epoch 2.0) |
| 5. Eval runner | `eval/run.py` → `results/` | ✅ done — 280 cells scored (96 result dirs) |
| 6. Merge grid | `merge/` → `merged/` | ✅ done — 55 PEFT + 36 sweep + 10 mergekit, all geometry-checked |
| 7. Analysis | `analysis/report.py` | ✅ done — CSVs, plots, summary.json, FINDINGS.md |
| 8. GGUF deploy | `deploy/build.py` | ✅ done — magnitude_prune__sql-math-translation-json-Q4_K_M.gguf (941 MB) |

## Verified data sources (probed live 2026-07-22, unauthenticated)

- ✅ `b-mc2/sql-create-context` — keys `answer / question / context`.
- ✅ `openai/gsm8k` (name=`main`) — keys `question / answer`. **NOTE:** config.data.sources.math
  says `hf_dataset: gsm8k`, but the bare id fails in datasets 5.0 — it must be
  **`openai/gsm8k`**. Handle in the data script (or fix config); flagged, not yet changed.
- ✅ `ai4bharat/samanantar` (name=`ta`, translation TRAIN) — `src`=English, `tgt`=Tamil.
  So for Ta→En: **input = `tgt`, output = `src`**. Huge dataset → use streaming + seeded take.
- ⛔ `facebook/flores` (translation EVAL) — **GATED**, needs an authenticated HF token
  with the FLORES license accepted. Old script-based mirrors are dead in datasets 5.0;
  the non-gated all-pairs parquet mirror is enormous (36 shards).

## RESOLVED — translation eval set

Settled as **option 2**: a disjoint, seed-deterministic slice of samanantar `ta`.
`data/prepare.py` never touches FLORES. Consequence for the paper (§VIII-C): the
eval set is same-distribution as train, so absolute chrF++ is optimistic. Every
model in the grid is scored on the same set, so the *relative* method comparison
is preserved — but absolute translation numbers must not be read as benchmark
results.

## Merge stage (§V) — how it works

`merge/grid.py` is pure enumeration (no torch): 6 pairs + 4 triples + 1 quad = 11
combos, x 5 methods = 55. `merge/build.py` does the work via PEFT
`add_weighted_adapter`, which supports all five of config's methods natively.

- **Cheap and GPU-free.** Merging is arithmetic on the rank-16 A/B factors
  (~71 MB each), not the 2.9 GB base. The base loads on CPU only because PEFT
  needs the module structure. Full 55-model grid = **~1 min**.
- **DARE is stochastic.** `dare_ties` / `dare_linear` drop entries at random, so
  every merge reseeds torch from `hash(seed, method, combo, weights)` before
  calling PEFT. Verified byte-reproducible across rebuilds, and independent of
  build order. The other three methods are deterministic.
- **Staged writes.** Output goes to `.<id>.partial` then moves into place, so an
  interrupted build never leaves a dir that `--resume` would count as finished.
- **Geometry gate.** Refuses to run unless all four specialists share identical
  `r / alpha / dropout / bias / target_modules`.
- `merged/` is gitignored (3.8 GB). Rebuild with `python -m merge.build`.

**Caveat to state in the paper:** PEFT's non-SVD `ties`/`dare_ties` elect signs on
the LoRA **A and B factors separately**, not on the composed delta
`(alpha/r)·B@A`. Standard PEFT behaviour, but §V currently describes it as acting
on the delta. Either reword, or switch to the `*_svd` variants. This caveat is
recorded in every `merged/*/merge_meta.json`.

Two merge parameters the paper does not specify are now pinned in config:
`merge.density: 0.5` (retained fraction, held identical across all pruning
methods so the comparison is not confounded) and `merge.majority_sign_method:
total`. Observed nonzero fractions: linear 100%, TIES 68.7%, DARE-TIES 93.8%.

## Eval runner (stage 5, complete)

`eval/run.py` scored 280 cells (96 model dirs × relevant tasks × 300 items each).
Per-item predictions stored in `results/<model_id>/<task>.preds.jsonl`; aggregate
scores in `results/<model_id>/<task>.json`. Supports `--resume` and `--shard k/N`.

## Key results

- **Best method:** `magnitude_prune` (mean retention 0.991, median 0.999)
- **Deployed model:** `magnitude_prune__sql-math-translation-json` at 0.960 mean retention
- **Fragile tasks:** SQL (most affected by merging), Translation (hurt by TIES specifically)
- **Robust tasks:** Math and JSON (essentially unaffected by any method)
- **Coefficient sweep:** Equal weights (0.5/0.5) beat skewed weights
- **Mergekit cross-check:** Pearson r=0.48, directional agreement but magnitude differences

See `FINDINGS.md` for the full recommendation report.

## All blockers resolved

- **`llama.cpp`** cloned and built → `llama.cpp/build/bin/Release/`
- **mergekit cross-check** completed on 10-model slice → `results_mergekit/`
- **Coefficient sweep** completed for top 2 methods (magnitude_prune, dare_ties)
- **Dashboard:** `index.html` — open in browser to view interactive analysis

## How to run things

```bash
.venv/bin/python -m eval.test_scorers        # scorer gate (must stay green)
.venv/bin/python scripts_download_model.py   # idempotent; re-pins if needed
```

Git repo on branch `main`. GPU present (`torch.cuda.is_available()` → True).
