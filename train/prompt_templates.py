"""Shared prompt template — THE load-bearing experimental control (spec §5).

ALL FOUR adapters are trained, and every model is evaluated, with an IDENTICAL
prompt structure. Only the per-task *instruction* string differs. If templates
differed structurally, a merged model could fail merely because it cannot resolve
conflicting output formats, and we would misattribute a formatting collision to
task interference.

Structure (Qwen chat template, single system message, instruction inside the user turn):

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    {instruction}

    {input}<|im_end|>
    <|im_start|>assistant
    {output}<|im_end|>

Loss is computed on the assistant turn ONLY (prompt tokens masked). To make that
masking robust across tokenizers we build the prompt and the full text separately
and mask by length — see `build_training_example`.
"""

from __future__ import annotations

SYSTEM_MESSAGE = "You are a helpful assistant."

# Qwen2.5 special tokens (kept explicit so the template is self-documenting and
# does not silently depend on tokenizer.apply_chat_template internals).
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def build_prompt(instruction: str, input_text: str) -> str:
    """Everything up to and including the assistant tag — the part we condition on.

    The trailing "<|im_start|>assistant\n" primes generation. At eval time this is
    exactly what we feed the model; at train time it is the region we mask out.
    """
    return (
        f"{IM_START}system\n{SYSTEM_MESSAGE}{IM_END}\n"
        f"{IM_START}user\n{instruction}\n\n{input_text}{IM_END}\n"
        f"{IM_START}assistant\n"
    )


def build_full_text(instruction: str, input_text: str, output_text: str) -> str:
    """Full training string: prompt + assistant answer + <|im_end|>.

    Note the assistant turn is closed with IM_END so the model learns to stop.
    """
    return f"{build_prompt(instruction, input_text)}{output_text}{IM_END}\n"


def build_training_example(tokenizer, instruction: str, input_text: str, output_text: str,
                           max_seq_len: int) -> dict:
    """Tokenize one example and produce labels with the prompt masked to -100.

    Returns dict with input_ids, attention_mask, labels (all lists, length <= max_seq_len).
    The assistant-turn tokens (answer + closing IM_END + newline) keep their token ids
    as labels; every prompt token is -100 so it contributes no loss.
    """
    prompt = build_prompt(instruction, input_text)
    full = build_full_text(instruction, input_text, output_text)

    # add_special_tokens=False: the chat template already carries all needed markers,
    # and Qwen2.5 does not prepend BOS. This keeps prompt_ids a true prefix of full_ids.
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]

    # Defensive: prompt must be a prefix of full. If a tokenizer ever violates this
    # (e.g. merges across the boundary), fall back to re-tokenizing separately.
    if full_ids[: len(prompt_ids)] != prompt_ids:
        answer_ids = tokenizer(output_text + IM_END + "\n", add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + answer_ids

    n_prompt = len(prompt_ids)
    labels = [-100] * n_prompt + full_ids[n_prompt:]

    # Truncate from the right; keep input_ids and labels aligned.
    input_ids = full_ids[:max_seq_len]
    labels = labels[:max_seq_len]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
