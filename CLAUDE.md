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
| 5. Eval runner | `eval/` | ⬜ todo → `results/` — **the bottleneck**, see below |
| 6. Merge grid | `merge/` → `merged/` | ✅ built — all 55 present (3.8 GB), distinct, geometry-checked |
| 7. Analysis | `analysis/` | ⬜ todo |
| 8. GGUF deploy | `deploy/` | ⬜ todo — Q4_K_M |

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

## Eval plan (stage 5, not yet built)

~60 models x 4 tasks x 300 items = **72k greedy generations, est. 20–40 h** on the
8 GB card. Design agreed:

- **Shard unit = one (model, task) cell** (300 gens, ~5–10 min). 240 cells total.
  `--shard k/N` slices the cell list; `--resume` skips finished cells.
- **Persist per-item predictions, aggregate late.** chrF++ is corpus-level and
  cannot be averaged across shards; JSON micro-F1 pools tp/fp/fn globally. Storing
  per-item records and calling `score_corpus` once at aggregation makes *any*
  sharding scheme exact.
- **Run the 5 control rows first** (20 cells, ~2 h). Specialist scores are the
  denominator of R_i = s_i(θ_M)/s_i(θ_i) — no retention number exists until they do.
- Add `--limit N` and time one cell per task before committing to a schedule; the
  20–40 h figure is extrapolation, not measurement. `math`/`json` allow 256 new
  tokens vs 128 for `sql`/`translation`, so expect ~2x spread by task.
- JSON scorer wants a `schema` in `extra`, but the eval JSONL embeds the schema
  inside the `input` text — the runner must parse it back out.

## Known blockers for later stages

- **`llama.cpp` absent** → stage 8 (GGUF Q4_K_M) cannot run until it is cloned/built.
- **mergekit cross-check** (~10 configs) merges *full checkpoints*, not adapters:
  ~3.1 GB each, **~31 GB** disk. 382 GB free, so fine, but it is not the cheap path
  that the PEFT grid is.
- **Coefficient sweep** is built by `merge/build.py --sweep --methods a,b`, but the
  "top 2 methods" are unknown until eval runs — deliberately not hardcoded.

SQL / math / json data prep is fully unblocked regardless of this choice.

## How to run things

```bash
.venv/bin/python -m eval.test_scorers        # scorer gate (must stay green)
.venv/bin/python scripts_download_model.py   # idempotent; re-pins if needed
```

Git repo on branch `main`. GPU present (`torch.cuda.is_available()` → True).
