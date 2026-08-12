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
| 4. Train LoRA adapters | `train/run.py` → `adapters/` | 🟡 2/4 — SQL (loss 0.019) + JSON (loss 7.5e-06) done, math/translation todo |
| 5. Eval runner | `eval/` | ⬜ todo → `results/` |
| 6. Merge grid | `merge/` | ⬜ todo — see config.merge (55 merged + 5 control) |
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

## OPEN DECISION — translation eval set (blocks only the translation eval, nothing else)

Was asked, not yet resolved. Pick one before building the translation eval:
1. **User provides HF token** (accept FLORES license, set `HF_TOKEN` / `hf auth login`),
   then use `facebook/flores` `tam_Taml-eng_Latn` devtest exactly — most faithful.
2. **Hold out a disjoint, seed-deterministic slice of samanantar ta** — non-gated,
   reproducible, but same-distribution as train (weaker than a clean benchmark).
3. **Another public Ta→En test set** — external like FLORES but unverified quality.

SQL / math / json data prep is fully unblocked regardless of this choice.

## How to run things

```bash
.venv/bin/python -m eval.test_scorers        # scorer gate (must stay green)
.venv/bin/python scripts_download_model.py   # idempotent; re-pins if needed
```

Git repo on branch `main`. GPU present (`torch.cuda.is_available()` → True).
