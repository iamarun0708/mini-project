# Systematic Study of Model Merging in Small Language Models

A controlled experiment to study **task interference vs. transfer** when merging
LoRA adapters trained on diverse tasks into a single small language model.

## Overview

We fine-tune four task-specialist LoRA adapters on a single pinned base model,
merge them using multiple weight-space methods, and measure how well the merged
models retain specialist performance while gaining cross-task generalization.

## Base Model

| Property | Value |
|----------|-------|
| Model | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Parameters | 1.5B |
| Revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (pinned) |
| Precision | bfloat16 (no quantization — avoids noise in merge study) |

## Tasks & Adapters

| Task | Description | Dataset | Train Size | Final Loss |
|------|-------------|---------|------------|------------|
| **SQL** | Text-to-SQL generation | [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) | 8,000 | 0.019 |
| **Math** | Step-by-step reasoning | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | 7,173 | 0.271 |
| **Translation** | Tamil → English | [ai4bharat/samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) (ta) | 8,000 | 1.84 |
| **JSON** | Schema-based extraction | Synthetic (10 schemas, seed 42) | 8,000 | 7.5e-06 |

All adapters share identical LoRA configuration (r=16, α=32, dropout=0.05) targeting
q/k/v/o/gate/up/down projections — a requirement for weight-space merging.

## Hardware

- **GPU**: NVIDIA GeForce RTX 4070 Laptop (8 GB VRAM)
- **Training**: batch=2, grad_accum=16, effective batch=32
- **Training time**: ~30 min per adapter

## Project Structure

```
.
├── config.yaml              # Single source of truth — all hyperparams & paths
├── common.py                # Config loader, seeding, overlap hashing
│
├── data/
│   ├── prepare.py           # Downloads HF datasets + generates synthetic JSON
│   └── processed/           # Train/eval JSONL splits (generated)
│
├── train/
│   ├── run.py               # LoRA fine-tuning script (--task sql|math|translation|json)
│   └── prompt_templates.py  # Shared Qwen chat-format prompt builder + loss masking
│
├── eval/
│   ├── scorers/             # Task-specific evaluation metrics
│   │   ├── sql.py           # AST equivalence via sqlglot
│   │   ├── math.py          # Final number extraction + numeric comparison
│   │   ├── translation.py   # chrF++ via sacrebleu (corpus-level)
│   │   └── json_extract.py  # Schema validation + per-field F1
│   └── test_scorers.py      # 24-check unit tests for all scorers
│
├── adapters/                # Trained LoRA adapter weights
│   ├── sql/
│   ├── math/
│   ├── translation/
│   └── json/
│
├── merge/                   # Merge scripts (planned)
├── analysis/                # Analysis & visualization (planned)
├── results/                 # Evaluation output (planned)
├── deploy/                  # GGUF quantized models (planned)
│
├── models/
│   └── qwen2.5-1.5b-instruct/  # Pinned base model snapshot
│
└── .venv/                   # Python virtual environment
```

## Pipeline

```mermaid
graph LR
    A[Download & Pin Model] --> B[Prepare Data]
    B --> C[Train 4 LoRA Adapters]
    C --> D[Evaluate Individual Adapters]
    D --> E[Merge Grid: 5 Methods × 11 Combos]
    E --> F[Evaluate Merged Models]
    F --> G[Analysis & Visualization]
    G --> H[GGUF Deployment]
```

| Stage | Status |
|-------|--------|
| Download & pin base model | ✅ Complete |
| Prepare datasets | ✅ Complete |
| Train all 4 adapters | ✅ Complete |
| Build evaluation runner | ⬜ Planned |
| Merge grid (55 merged + 5 controls) | ⬜ Planned |
| Analysis & visualization | ⬜ Planned |
| GGUF deployment (Q4_K_M) | ⬜ Planned |

## Merge Methods

Five weight-space merging strategies will be evaluated:

1. **Linear** — weighted average of adapter parameters
2. **TIES** — trim, elect sign, merge (resolves sign conflicts)
3. **DARE-TIES** — drop and rescale + TIES
4. **DARE-Linear** — drop and rescale + linear average
5. **Magnitude Prune** — prune low-magnitude parameters before merging

Each method is applied to 11 adapter combinations (6 pairs + 4 triples + 1 quad),
producing 55 merged models + 5 control rows (base + 4 individual adapters).

## Evaluation Metrics

| Task | Metric | Implementation |
|------|--------|----------------|
| SQL | AST equivalence | sqlglot parse + normalize |
| Math | Exact match (numeric) | Final number extraction |
| Translation | chrF++ | sacrebleu corpus-level |
| JSON | Per-field F1 | Schema validation + field matching |

All evaluation uses **greedy decoding** (`do_sample: false`) for reproducibility.

## Reproducibility

- **Seed**: 42, applied globally (torch, numpy, random, transformers)
- **Model revision**: Pinned to exact HuggingFace commit
- **Config-driven**: All hyperparameters in `config.yaml`, never hardcoded
- **Identical LoRA config**: Same rank, alpha, dropout, and target modules across all adapters

## Quick Start

```bash
# 1. Setup environment
bash scripts_install.sh

# 2. Download and pin model
.venv/bin/python scripts_download_model.py

# 3. Prepare datasets
.venv/bin/python -m data.prepare

# 4. Train adapters (one at a time for 8GB VRAM)
.venv/bin/python -m train.run --task sql
.venv/bin/python -m train.run --task math
.venv/bin/python -m train.run --task translation
.venv/bin/python -m train.run --task json

# 5. Verify scorers
.venv/bin/python -m eval.test_scorers
```

## License

This is an academic research project / mini-project for systematic study purposes.
