# AGENTS.md — Antigravity IDE onboarding for this project

Read this FIRST before crawling the tree. This gives you full context to continue
work without re-reading every file.

## What this project is

**Systematic study of model merging in small language models.** Train four
task-specialist LoRA adapters on ONE pinned base model, merge them with several
weight-space methods, and measure task interference vs. transfer.

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`, pinned to commit
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Downloaded to `models/qwen2.5-1.5b-instruct/`.
- **Four tasks:** `sql`, `math`, `translation` (Tamil→English), `json` (extraction).
- **Hardware:** RTX 4070 Laptop, **8 GB VRAM**. Batch=2, grad_accum=16, effective batch=32.

## Single source of truth

`config.yaml` — all paths, seeds, hyperparameters, task instructions, merge grid,
the pinned model revision. `common.py` loads it and applies global seeding (seed 42).

## Key architectural decisions

- **Prompt template is the load-bearing control** (`train/prompt_templates.py`):
  identical structure for all adapters + all eval; only per-task `instruction` differs.
  Loss is masked to the assistant turn only.
- **Eval is greedy** (`do_sample: false`) everywhere — reproducibility over peak scores.
- **LoRA config identical across all adapters**: r=16, alpha=32, dropout=0.05,
  targeting q/k/v/o/gate/up/down projections. This is required for merge methods.
- **Translation eval:** Uses held-out samanantar slice (FLORES is gated, no HF token).
  Config still says FLORES; the `data/prepare.py` script handles the actual holdout.
- **Math note:** `config.yaml` says `hf_dataset: gsm8k` but bare name fails in
  datasets 5.0 — `data/prepare.py` uses `openai/gsm8k`. GSM8K has only 7,473 train
  examples (vs 8,000 target), so math trains on 7,173 with 300 eval.

## Pipeline status

| Stage | Location | Status |
|-------|----------|--------|
| 0. Install stack | `scripts_install.sh` | ✅ done |
| 1. Download + pin base model | `scripts_download_model.py` | ✅ done |
| 2. Scorers | `eval/scorers/{sql,math,translation,json_extract}.py` | ✅ done (24/24 tests pass) |
| 3. Data preparation | `data/prepare.py` | ✅ done — all 4 datasets in `data/processed/` |
| 4a. Train SQL adapter | `train/run.py --task sql` | ✅ done → `adapters/sql/` (loss 0.019) |
| 4b. Train math adapter | `train/run.py --task math` | ✅ done → `adapters/math/` (loss 0.271) |
| 4c. Train translation adapter | `train/run.py --task translation` | ✅ done → `adapters/translation/` (loss 1.84) |
| 4d. Train json adapter | `train/run.py --task json` | ✅ done → `adapters/json/` (loss 7.5e-06) |
| 5. Eval runner | `eval/` | ⬜ todo — scorers exist, need runner script |
| 6. Merge grid | `merge/` | ⬜ todo — 55 merged + 5 control models |
| 7. Analysis | `analysis/` | ⬜ todo |
| 8. GGUF deploy | `deploy/` | ⬜ todo — Q4_K_M |

## Data files (in `data/processed/`)

| Task | Train file | Train count | Eval file | Eval count |
|------|-----------|-------------|-----------|------------|
| SQL | `sql_train.jsonl` | 8,000 | `sql_eval.jsonl` | 300 |
| Math | `math_train.jsonl` | 7,173 | `math_eval.jsonl` | 300 |
| Translation | `translation_train.jsonl` | 8,000 | `translation_eval.jsonl` | 300 |
| JSON | `json_train.jsonl` | 8,000 | `json_eval.jsonl` | 300 |

Each JSONL record has: `{"task", "instruction", "input", "output"}`

## File map (what each file does)

### Root
- `config.yaml` — **READ THIS**: all hyperparams, paths, model revision, merge grid
- `common.py` — config loader, seeding, overlap hashing
- `README.md` — project overview, pipeline status, quick start guide
- `scripts_install.sh` — venv setup (already run)
- `scripts_download_model.py` — model download + revision pinning (already run)
- `.gitignore` — excludes models/, adapters/, data/processed/, .venv/, __pycache__, *.gguf, llama.cpp/
- `.antigravityignore` — keeps heavy/generated dirs out of Antigravity IDE context
- `.agents/AGENTS.md` — this file; onboarding doc for AI agents

### `train/`
- `prompt_templates.py` — shared Qwen chat-format prompt builder, loss masking
- `run.py` — LoRA fine-tuning script. Usage: `--task {sql,math,translation,json}` or all

### `data/`
- `prepare.py` — downloads HF datasets + generates synthetic JSON, splits train/eval

### `eval/`
- `scorers/sql.py` — AST equivalence via sqlglot
- `scorers/math.py` — final number extraction + numeric comparison
- `scorers/translation.py` — chrF++ via sacrebleu (corpus-level)
- `scorers/json_extract.py` — schema validation + per-field F1
- `scorers/_util.py` — shared cleaning (fence stripping, im_end removal)
- `test_scorers.py` — 24-check gate, run with `.venv/bin/python -m eval.test_scorers`

### `adapters/`
- `sql/` — ✅ trained adapter (adapter_config.json + adapter_model.safetensors)
- `math/` — ✅ trained adapter (adapter_config.json + adapter_model.safetensors)
- `translation/` — ✅ trained adapter (adapter_config.json + adapter_model.safetensors)
- `json/` — ✅ trained adapter (adapter_config.json + adapter_model.safetensors)

### Empty dirs (to be populated)
- `merge/` — merge scripts
- `analysis/` — analysis scripts
- `results/` — eval output
- `deploy/` — GGUF models

## How to continue

All 4 adapters are trained. The remaining stages are:

1. **Build `eval/run.py`** — evaluation runner using existing scorers
2. **Build `merge/run.py`** — merge grid (5 methods × 11 combos = 55 merged + 5 controls)
3. **Build `analysis/`** — results tables + plots
4. **Build `deploy/`** — GGUF conversion (Q4_K_M)

```bash
# Verify scorers still pass:
.venv/bin/python -m eval.test_scorers
```

## Conventions

- **venv:** Always use `.venv/bin/python`. Never `pip install` outside it.
- **Config-driven:** All hyperparams come from `config.yaml`, never hardcoded.
- **Seed 42:** Applied everywhere via `common.set_all_seeds()`.
- **No git repo.** GPU present (`torch.cuda.is_available()` → True).
