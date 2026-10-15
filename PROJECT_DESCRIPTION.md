# Systematic Study of Model Merging in Small Language Models (SLMs)
## Complete Project Description, Technical Architecture, Techniques & Tooling Guide

---

## 1. Executive Summary

This project presents a **systematic, empirical study of weight-space model merging in Small Language Models (SLMs)**. While large language models (70B+) possess immense parameter redundancy, small models ($\le$ 3B parameters) are constrained by tight capacity budgets, making multi-task weight-space fusion susceptible to destructive parameter interference and cognitive collapse.

The core objective of this research is to investigate:
1. **Feasibility**: Can four distinct, high-intensity task specialists (SQL synthesis, Math reasoning, Tamil-to-English translation, and JSON structured extraction) be fine-tuned as independent LoRA adapters on a single pinned 1.5B base model and subsequently merged in weight space without further training?
2. **Interference vs. Transfer**: Which merging techniques preserve individual specialist capabilities, which tasks suffer from catastrophic interference, and can synergistic cross-task transfer occur in a compact 1.5B architecture?
3. **Deployment**: Can the winning multi-task merged model be compiled, quantized, and deployed on consumer edge hardware (RTX 4070 Laptop / RTX A2000 with 8–12 GB VRAM) as a sub-1GB high-speed inference artifact?

### Key Breakthrough Findings
- **The Winner**: **Magnitude Pruning (`magnitude_prune`)** is the dominant merge technique, achieving **99.1% mean task retention** across all 280 experimental cells.
- **Positive Synergistic Transfer**: Merging the Math specialist with SQL and Translation unlocked **+5.2% higher accuracy on GSM8K math reasoning** than the standalone Math adapter itself.
- **Task Fragility**: SQL generation is the most fragile task due to rigid Abstract Syntax Tree (AST) constraints. Unpruned linear averaging degraded 4-way SQL performance down to 71.8%, whereas Magnitude Pruning maintained 86.8%–100% retention.
- **Edge Deployment**: The final 4-task model was materialized and quantized to **Q4_K_M GGUF (941 MB)**, executing locally at **~68 tokens/second** using **1.2 GB inference VRAM**.

---

## 2. Experimental Design & Controlled Methodology

To guarantee scientific validity, the experimental pipeline was designed around strict load-bearing controls:

```
                               ┌────────────────────────────────────────────────────────┐
                               │       Base Model: Qwen2.5-1.5B-Instruct (bfloat16)      │
                               │        Commit: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306│
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                   ┌───────────────────────┬───────────────┴───────────────┬───────────────────────┐
                   ▼                       ▼                               ▼                       ▼
         ┌──────────────────┐    ┌──────────────────┐            ┌──────────────────┐    ┌──────────────────┐
         │   SQL Adapter    │    │   Math Adapter   │            │Translation Adapt.│    │   JSON Adapter   │
         │  b-mc2 / Spider  │    │  OpenAI GSM8K    │            │    Samanantar    │    │ Synthetic Schema │
         │  Loss: 0.019     │    │  Loss: 0.271     │            │    Loss: 1.84    │    │ Loss: 7.5e-6     │
         └─────────┬────────┘    └─────────┬────────┘            └─────────┬────────┘    └─────────┬────────┘
                   │                       │                               │                       │
                   └───────────────────────┴───────────────┬───────────────┴───────────────────────┘
                                                           │
                                                           ▼
                                         ┌──────────────────────────────────┐
                                         │       Weight-Space Merging       │
                                         │  (PEFT Engine + Mergekit Cross)  │
                                         │  6 Pairs + 4 Triples + 1 Quad    │
                                         │  x 5 Pruning & Linear Methods    │
                                         │  = 55 Merged Models + 36 Sweeps  │
                                         └─────────────────┬────────────────┘
                                                           │
                                                           ▼
                                         ┌──────────────────────────────────┐
                                         │  Deterministic Greedy Evaluation │
                                         │   280 Scored Cells (AST, Exact   │
                                         │   Number, chrF++, Schema F1)     │
                                         └─────────────────┬────────────────┘
                                                           │
                                                           ▼
                                         ┌──────────────────────────────────┐
                                         │   Q4_K_M GGUF Edge Deployment    │
                                         │    FastAPI + Llama-Server Studio │
                                         └──────────────────────────────────┘
```

### A. The Single Source of Truth (`config.yaml`)
All hyperparameters, paths, random seeds, pinned revisions, merge grids, and task instructions are centralized in `config.yaml`. Global seed `42` is enforced across PyTorch, NumPy, Python standard random, and Hugging Face Transformers via `common.set_all_seeds()`.

### B. Base Architecture & Uniform LoRA Configuration
- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Revision Commit**: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (strictly pinned to prevent upstream drifting)
- **Precision**: `bfloat16` across all LoRA training (no 4-bit QLoRA quantization noise in weight deltas)
- **LoRA Parameters**: Identical across all 4 specialist adapters to guarantee matching weight-space subspace rank:
  - Rank ($r$): `16`
  - Scaling factor ($\alpha$): `32`
  - Dropout: `0.05`
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  - Bias: `none`

### C. Prompt Template as an Experimental Control
All training and evaluations use an identical Qwen Chat structure (`train/prompt_templates.py`), with loss computed on the assistant turn only:
```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}

{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

### D. The Four Specialized Tasks

| Task | Dataset | Samples | Metric & Tool | Base Model Score | Specialist Score | Lift Achieved |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SQL Synthesis** | `b-mc2/sql-create-context` (Spider schema subset) | 8,000 train<br>300 eval | AST Equivalence via `sqlglot` | 0.027 (2.7%) | 0.757 (75.7%) | **28x improvement** |
| **Math Reasoning** | `openai/gsm8k` (grade school math) | 7,173 train<br>300 eval | Strict Final Numeric Match | 0.563 (56.3%) | 0.643 (64.3%) | **+14.2% lift** |
| **Ta→En Translation** | `ai4bharat/samanantar` (Tamil subset) | 8,000 train<br>300 eval | Corpus-level chrF++ via `sacrebleu` | 18.13 chrF++ | 23.81 chrF++ | **+31.4% lift** |
| **JSON Extraction** | Synthetic deterministic schema generation (10 schemas) | 8,000 train<br>300 eval | Schema Validation + Field F1 | 0.952 (95.2%) | 1.000 (100.0%) | **+5.0% lift (perfect)** |

---

## 3. Merging Techniques Explained

This study evaluated eight distinct weight-space merging techniques and parameter tuning sweeps:

### 1. Magnitude Pruning (`magnitude_prune`) — **Rank #1 Winner**
- **Intuition**: Fine-tuning produces small, noisy weight shifts that have low individual utility but cause constructive/destructive interference when added across multiple adapters. Trimming the lowest-magnitude deltas isolates the true task-specific updates.
- **Formula**:
  $$\Delta W_{\text{merged}} = \sum_{i=1}^N \lambda_i \cdot \left[ \Delta W_i \odot \mathbb{I}\left(|\Delta W_i| \ge \tau_{\text{density}}\right) \right]$$
  where $\tau_{\text{density}}$ is the threshold retaining the top 50% absolute values ($\text{density} = 0.5$).
- **Performance**: **0.9905 mean retention**. Preserves 99.1% across all domains; delivers **+5.2% reasoning transfer on GSM8K**.

### 2. DARE + TIES (`dare_ties`) — **Rank #2**
- **Intuition**: Combines Drop And REscale (DARE) with Trimming, Electing Sign, and Merging Disjoint Elements (TIES). It uses Bernoulli dropout ($p=0.5$), rescales survivors by $1/(1-p)$, and enforces sign consensus across models.
- **Formula**:
  $$\Delta W_{\text{merged}} = \text{SignElect}\left( \left\{ \frac{\Delta W_i \odot \text{Bernoulli}(1-p)}{1-p} \right\} \right)$$
- **Performance**: **0.9711 mean retention**. Excellent on Math (+6.3% transfer), but sign voting degrades Tamil translation down to 82.6%.

### 3. Linear Delta Averaging (`linear`) — **Rank #3**
- **Intuition**: Direct uniform or weighted arithmetic addition of adapter parameter deltas onto the base model.
- **Formula**:
  $$\Delta W_{\text{merged}} = \sum_{i=1}^N \lambda_i \Delta W_i, \quad W_{\text{merged}} = W_{\text{base}} + \Delta W_{\text{merged}}$$
- **Performance**: **0.9681 mean retention**. Near-perfect on 2-way pairs (99.0%), but exhibits catastrophic interference in 4-way merges on syntax-heavy SQL (drops to 71.8%).

### 4. DARE Linear (`dare_linear`) — **Rank #4**
- **Intuition**: Applies Bernoulli random masking and rescaling without sign consensus voting before linear combination.
- **Formula**:
  $$\Delta W_{\text{merged}} = \sum_{i=1}^N \lambda_i \cdot \left[ \frac{\Delta W_i \odot \text{Bernoulli}(1-p)}{1-p} \right]$$
- **Performance**: **0.9596 mean retention**. Unmitigated gradient direction collisions cause severe SQL degradation (66.1% retention).

### 5. TIES-Merging (`ties`) — **Rank #5**
- **Intuition**: Trims the bottom 80% parameter deltas ($\text{density}=0.2$), determines a majority sign vector per coordinate, and averages disjoint parameters matching the elected sign.
- **Formula**:
  $$\Delta W_{\text{merged}} = \text{MajoritySign} \odot \text{DisjointMean}\left(\text{Trim}_{0.2}(\Delta W_i)\right)$$
- **Performance**: **0.9433 mean retention**. Over-pruning damages translation vocabulary retention (83.5%).

### 6. Task Arithmetic via Mergekit (`task_arithmetic`)
- **Intuition**: Independent cross-check merging full checkpoint safetensors tensors on disk without LoRA rank factorization.
- **Formula**:
  $$W_{\text{merged}} = W_{\text{base}} + \sum_{i=1}^N \lambda_i (W_i - W_{\text{base}})$$
- **Performance**: **0.9920 mean retention**, corroborating PEFT-based delta findings.

### 7. Mergekit TIES (`mergekit_ties`)
- **Intuition**: Full-rank checkpoint implementation of TIES executed across unquantized FP16 checkpoints via Mergekit.
- **Performance**: **0.9850 mean retention**.

### 8. Asymmetric Coefficient Sweeps (`coefficient_sweep`)
- **Intuition**: Evaluates pair trade-offs under asymmetric weight distributions ($\lambda \in \{[0.3, 0.7], [0.5, 0.5], [0.7, 0.3]\}$). Demonstrates how prioritizing one task shields it from interference while revealing Pareto-optimal frontiers.

---

## 4. Comprehensive Tools & Technology Stack

The project integrates production-grade open-source machine learning, systems programming, and web technologies:

### A. Deep Learning & Model Training Stack
| Tool / Library | Version / Pin | Purpose in Project |
| :--- | :--- | :--- |
| **Python** | 3.12 (in `.venv`) | Primary runtime environment |
| **PyTorch** | 2.5.1+cu124 | GPU acceleration, tensor computation, and bfloat16 mixed precision |
| **Transformers** | 4.49.0 | Model architecture loading, tokenization, generation pipelines |
| **PEFT** | 0.14.0 | LoRA adapter injection, parameter tracking, weight delta extraction |
| **Accelerate** | 1.4.0 | Multi-device memory budgeting and gradient accumulation management |
| **Datasets** | 3.3.2 | Downloading, slicing, caching, and preprocessing Hugging Face datasets |

### B. Evaluation, AST Parsing & Metrics Stack
| Tool / Library | Purpose in Project |
| :--- | :--- |
| **sqlglot** | Abstract Syntax Tree (AST) parser used in `eval/scorers/sql.py`. Normalizes column orders, whitespace, semicolons, and case to evaluate true semantic SQL query equivalence. |
| **sacrebleu** | Standard corpus-level `chrF++` scorer used in `eval/scorers/translation.py` for Tamil-to-English translation. |
| **jsonschema** | Draft 7 JSON schema validation in `eval/scorers/json_extract.py` with custom per-field micro-F1 scoring. |
| **Regex / Numeric Matcher** | Custom parser in `eval/scorers/math.py` that extracts final answers following step-by-step reasoning blocks and tolerates currency symbols, commas, and float-integer equivalence ($42.0 \equiv 42$). |

### C. Model Merging & Quantization Stack
| Tool / Library | Purpose in Project |
| :--- | :--- |
| **Custom PEFT Merge Engine** | Developed in `merge/build.py`. Implements pure vectorized tensor operations for `magnitude_prune`, `ties`, `dare_ties`, `dare_linear`, and `linear`. |
| **Mergekit** | External merge library utilized in `merge/mergekit_crosscheck.py` to perform independent full-checkpoint merges. |
| **llama.cpp** | Compiled from source (`llama.cpp/build/bin/Release/`). Provides high-performance C/C++ GGUF conversion, quantization, and CUDA GPU offloading (`-ngl 99`). |
| **llama-server.exe** | High-throughput local OpenAI-compatible inference server running on port `8090`. |
| **llama-cli.exe** | Terminal-based direct GGUF inference client. |
| **gguf (Python SDK)** | GGUF binary reader used for metadata inspection, tensor verification, and vocabulary token validation. |

### D. Full-Stack Web & Serving Architecture
| Component | Technology | Description |
| :--- | :--- | :--- |
| **Backend Framework** | **FastAPI** | High-performance asynchronous REST API backend in `server.py` running on `http://127.0.0.1:8000`. |
| **ASGI Server** | **Uvicorn** | Production-ready ASGI server hosting the FastAPI application. |
| **System Telemetry** | **psutil** & **torch.cuda** | Hardware probing for GPU name, allocated/free VRAM, system RAM, and CPU core counts. |
| **Process Manager** | **Python subprocess** | Spawns, monitors, and restarts underlying `llama-server.exe` instances. |
| **Frontend Architecture** | **Vanilla HTML5 / Modern CSS** | Clean glassmorphic design system with curated dark palette, micro-animations, and responsive layouts in `index.html`. |
| **Data Visualizations** | **Chart.js** | Interactive radar charts, bar charts, and arity trend lines embedded in the dashboard. |

---

## 5. Repository Structure & Artifact Map

```
c:\Users\12403\Pictures\mini-project\
│
├── config.yaml                                # Single source of truth: all paths, hyperparams & tasks
├── common.py                                  # Config loader, global seeding (seed 42), overlap hashing
├── server.py                                  # FastAPI backend serving studio APIs and llama.cpp proxy
├── index.html                                 # Interactive 3-stage research & chat dashboard
│
├── train/
│   ├── run.py                                 # LoRA fine-tuning runner (--task {sql,math,translation,json})
│   └── prompt_templates.py                    # Shared Qwen chat-format prompt builder + loss masking
│
├── data/
│   ├── prepare.py                             # Downloads HF datasets and generates synthetic JSON
│   └── processed/                             # Train (8,000 / 7,173) and eval (300) JSONL datasets
│
├── adapters/                                  # Trained specialist LoRA weights (adapter_model.safetensors)
│   ├── sql/                                   # Loss 0.019 (Spider)
│   ├── math/                                  # Loss 0.271 (GSM8K)
│   ├── translation/                           # Loss 1.840 (Samanantar)
│   └── json/                                  # Loss 7.5e-6 (Synthetic Schema)
│
├── merge/
│   ├── grid.py                                # Combo enumerator (6 pairs + 4 triples + 1 quad = 11 combos)
│   ├── build.py                               # PEFT merge engine (55 models + 36 sweeps)
│   ├── mergekit_crosscheck.py                 # Full checkpoint mergekit script
│   └── mergekit_eval.py                       # Mergekit model evaluator
│
├── eval/
│   ├── run.py                                 # Evaluation runner (280 cells scored, greedy decoding)
│   ├── test_scorers.py                        # 24-check unit test suite for all scorers
│   └── scorers/
│       ├── _util.py                           # Output sanitization, fence stripping, delimiter removal
│       ├── sql.py                             # AST parsing via sqlglot
│       ├── math.py                            # Number extraction & comparison
│       ├── translation.py                     # chrF++ via sacrebleu
│       └── json_extract.py                    # Schema validation + Field F1
│
├── analysis/
│   ├── report.py                              # Generates summary.json and primary analysis CSVs
│   ├── sweep_report.py                        # Coefficient sweep analysis
│   ├── summary.json                           # Machine-readable key findings
│   ├── retention.csv                          # Primary cell-by-cell retention table
│   ├── by_method.csv                          # Method ranking summary
│   ├── by_arity.csv                           # Retention degradation by arity (2, 3, 4 tasks)
│   ├── mergekit_vs_peft.csv                   # Checkpoint vs PEFT cross-framework data
│   └── plots/                                 # Generated matplotlib charts
│
├── deploy/
│   ├── build.py                               # GGUF compilation & quantization script
│   └── magnitude_prune__sql-math-translation-json-Q4_K_M.gguf  # Deployed model (941 MB)
│
├── results/                                   # Evaluation logs for all 96 model configurations
├── models/
│   └── qwen2.5-1.5b-instruct/                 # Pinned base model checkpoint
└── llama.cpp/                                 # Precompiled llama.cpp binaries (Release build)
```

---

## 6. Real-Time Chat Studio & Task Routing Engine

In a 1.5B weight-merged model, prompts lacking clear task instructions can cause adapter weight attractor collision (e.g. unprompted software engineering questions activating strong SQL weights). 

To resolve this **without modifying weights or retraining**, `server.py` implements an **Inference-Time Task Intent Classifier & Context Shield**:

1. **Automatic Intent Classifier (`classify_intent`)**:
   - Inspects user input tokens and patterns:
     - `general_qa`: Questions asking for debugging, code explanations, or runtime errors (*"how to resolve runtime error...", "explain..."*).
     - `sql`: Explicit query generation requests (*"write a sql query...", "select from..."*).
     - `math`: Arithmetic, word problems, or equations with numeric calculations.
     - `translation`: Tamil Unicode script detection (`\u0B80 - \u0BFF`).
     - `json`: Structured extraction requests.
2. **Training-Aligned Prompt Injection**:
   - Seamlessly injects the exact benchmark instruction header used during training, priming the merged attention heads into the correct subspace.
3. **In-Context Echo Suppression**:
   - Filters out repetitive table outputs from earlier turns when the user switches to conceptual software engineering questions, preventing conversational memory poisoning.

---

## 7. AI Model Parameter Predictor & Single-Method Merging Advisor

To eliminate guesswork when applying model merging across different architectures, the studio includes an **AI Parameter Predictor & Merge Synthesis Advisor** (accessible via the `🎯 Select & Merge Advisor` tab in the header).

### Safe Two-Phase Workflow (Inspection &rarr; Approval &rarr; Targeted Merge)

```
1. Model & Tasks Selection
   (Qwen2.5-1.5B, Llama-3.2-1B/3B, Phi-3.5-mini, Gemma-2-2B, or Custom HF/Local Path)
                 │
                 ▼
2. "Predict Parameters & Generate Pre-Merge Report"
   ├─► Inspects parameters: layers, hidden dim, attention heads, context length
   ├─► Probes hardware: dedicated GPU VRAM, allocated memory, compute budget
   ├─► Formulates Fine-Tuning Recipe: LoRA (bfloat16) vs QLoRA, rank r=16, alpha=32, target projections
   ├─► Formulates Merging Recipe: Recommends magnitude_prune (density 0.50) with mathematical justification
   └─► Predicts Task Retention Matrix across all 8 merging methods
                 │
                 ▼
3. User Review & Approval
   (User reviews feasibility report and approves the specific single method to execute)
                 │
                 ▼
4. "Confirm & Merge Selected Model"
   └─► Executes weight-space merge EXCLUSIVELY on the single approved method
   └─► Generates artifact metadata and enables instant testing in Chat Studio
```

---

## 8. How to Run & Verify the Project

### 1. Verify Scorer Integrity (24/24 Test Suite)
```powershell
.venv\Scripts\python -m eval.test_scorers
```
*Expected Output: `ALL SCORER CHECKS PASSED`*

### 2. Launch the Full-Stack Studio Backend
```powershell
.venv\Scripts\python server.py
```
*Starts FastAPI on `http://127.0.0.1:8000`, automatically managing `llama-server.exe` on port `8090`.*

### 3. Open the Interactive Research Dashboard
Open your browser at:
```
http://127.0.0.1:8000/
```
From the dashboard, you can:
- **Scan Local Models**: Discover pre-downloaded weights on your drive.
- **Probe Hardware Specs**: Check GPU VRAM and compute suitability budgets.
- **Simulate Any Merge**: Compare retention rates, radar plots, and formulas across all 8 merging algorithms.
- **Interactive Chat**: Test multi-turn conversations with real-time intent routing tags (`🎯 SQL`, `🎯 Math`, `🎯 General Q&A`).

### 4. Direct CLI Inference with GGUF Artifact
```powershell
.\llama.cpp\build\bin\Release\llama-cli.exe -m deploy\magnitude_prune__sql-math-translation-json-Q4_K_M.gguf -p "<|im_start|>system`nYou are a helpful assistant.<|im_end|>`n<|im_start|>user`nSolve the following math problem. Think step by step, then give the final numeric answer.`n`nIf a box has 24 apples and 6 are red, what percentage are red?<|im_end|>`n<|im_start|>assistant`n" -ngl 99
```

---

## 8. Summary of Results Matrix

```
Method Ranking by Mean Task Retention (280 Empirical Cells):
============================================================
1. magnitude_prune : 0.9905  [Winner — Synergistic +5.2% Math Transfer]
2. dare_ties       : 0.9711  [Strong Math (+6.3%), Lower Translation (82.6%)]
3. linear          : 0.9681  [High on 2-way (99.0%), Degrades on 4-way SQL (71.8%)]
4. dare_linear     : 0.9596  [Gradient collision on exact syntax]
5. ties            : 0.9433  [Over-pruning damages vocabulary retention]
```
