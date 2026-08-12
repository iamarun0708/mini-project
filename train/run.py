"""LoRA fine-tuning for task-specialist adapters (spec §11, stage 4).

Trains one LoRA adapter per task on the pinned Qwen2.5-1.5B-Instruct base model.
All hyperparameters come from config.yaml. Uses prompt_templates.build_training_example
for tokenization with loss masking (assistant turn only).

Run:
  .venv/bin/python -m train.run                # all 4 tasks sequentially
  .venv/bin/python -m train.run --task sql      # single task
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, set_all_seeds, model_revision  # noqa: E402
from train.prompt_templates import build_training_example  # noqa: E402

ALL_TASKS = ["sql", "math", "translation", "json"]


def load_task_data(processed_dir: Path, task: str) -> list[dict]:
    """Load JSONL training data for a task."""
    path = processed_dir / f"{task}_train.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found: {path}\n"
            f"Run: .venv/bin/python -m data.prepare"
        )
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  loaded {len(records):,} training examples from {path}")
    return records


def tokenize_dataset(
    records: list[dict],
    tokenizer,
    max_seq_len: int,
) -> Dataset:
    """Tokenize all records using the shared prompt template with loss masking."""
    tokenized = []
    skipped = 0
    for rec in records:
        example = build_training_example(
            tokenizer=tokenizer,
            instruction=rec["instruction"],
            input_text=rec["input"],
            output_text=rec["output"],
            max_seq_len=max_seq_len,
        )
        # Skip examples where the entire sequence is prompt (no trainable tokens)
        if all(l == -100 for l in example["labels"]):
            skipped += 1
            continue
        tokenized.append(example)

    if skipped > 0:
        print(f"  skipped {skipped} examples (output truncated away)")
    print(f"  tokenized {len(tokenized):,} examples")

    return Dataset.from_dict({
        "input_ids": [t["input_ids"] for t in tokenized],
        "attention_mask": [t["attention_mask"] for t in tokenized],
        "labels": [t["labels"] for t in tokenized],
    })


def train_adapter(task: str, cfg: dict) -> None:
    """Train a single LoRA adapter for the given task."""
    print(f"\n{'='*60}")
    print(f"  Training adapter: {task}")
    print(f"{'='*60}")

    seed = cfg["seed"]
    set_all_seeds(seed)

    model_dir = cfg["model"]["local_dir"]
    revision = model_revision(cfg)
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    adapters_dir = Path(cfg["merge"]["adapters_dir"])
    output_dir = adapters_dir / task

    # 1. Load tokenizer
    print(f"\n[1/5] Loading tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load and tokenize data
    print(f"\n[2/5] Loading and tokenizing {task} data...")
    records = load_task_data(processed_dir, task)
    dataset = tokenize_dataset(records, tokenizer, train_cfg["max_seq_len"])

    # 3. Load base model
    print(f"\n[3/5] Loading base model (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    # 4. Configure and apply LoRA
    print(f"\n[4/5] Applying LoRA (r={lora_cfg['r']}, alpha={lora_cfg['alpha']})...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Train
    print(f"\n[5/5] Training...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["epochs"],
        learning_rate=train_cfg["lr"],
        lr_scheduler_type=train_cfg["scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        seed=seed,
        report_to="none",  # no wandb/tensorboard
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        optim="adamw_torch",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # Save final adapter
    print(f"\n  Saving adapter to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Verify saved files
    expected_files = ["adapter_config.json"]
    for f in expected_files:
        assert (output_dir / f).exists(), f"Missing: {output_dir / f}"
    print(f"  ✅ Adapter saved: {output_dir}")

    # Free GPU memory
    del model, trainer, dataset
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU memory freed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA adapters")
    parser.add_argument("--task", type=str, choices=ALL_TASKS, default=None,
                        help="Train a single task (default: all tasks)")
    args = parser.parse_args()

    cfg = load_config()
    tasks = [args.task] if args.task else ALL_TASKS

    print(f"Base model: {cfg['model']['repo_id']}")
    print(f"Revision:   {model_revision(cfg)}")
    print(f"Tasks:      {tasks}")
    print(f"GPU:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    print(f"VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
          if torch.cuda.is_available() else "")

    for task in tasks:
        train_adapter(task, cfg)

    print(f"\n🎉 Training complete for: {tasks}")


if __name__ == "__main__":
    main()
