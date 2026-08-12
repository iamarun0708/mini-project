"""Shared helpers: config loading, global seeding, revision stamping.

Imported by every script so that (a) there is one source of truth for hyperparameters
and (b) reproducibility rules from spec §10 are applied identically everywhere.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    with open(path or CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def set_all_seeds(seed: int) -> None:
    """Seed random, numpy, torch, and transformers (spec §10)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass


def model_revision(cfg: dict) -> str:
    """Return the pinned base-model commit hash, or a loud placeholder."""
    return cfg.get("model", {}).get("revision", "UNKNOWN")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_for_overlap(text: str) -> str:
    """Canonical form for train/eval overlap hashing (spec §10)."""
    return " ".join(text.lower().split())


def overlap_hash(text: str) -> str:
    return sha256_text(normalize_for_overlap(text))
