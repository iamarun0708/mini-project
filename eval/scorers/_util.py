"""Shared cleaning helpers for scorers.

Models trained with our template should emit bare answers, but at eval time
(especially for the base model and degraded merges) outputs may be wrapped in
markdown fences or prefixed with chatter. Cleaning is applied uniformly so that
formatting noise is not scored as a task error where the spec says it shouldn't be.
"""

from __future__ import annotations

import re

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """If the text contains a fenced block, return the first block's contents.
    Otherwise return the text unchanged. Trims surrounding whitespace."""
    if text is None:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def strip_im_end(text: str) -> str:
    """Remove a trailing Qwen assistant-turn terminator if the model echoes it."""
    return text.replace("<|im_end|>", "").strip()


def clean(text: str) -> str:
    return strip_im_end(strip_code_fence(text or ""))
