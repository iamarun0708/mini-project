"""Download the base model at a resolved commit and pin that commit in config.yaml.

Spec: HF repos update silently; if adapters train against one snapshot and merge
against another the results are meaningless. So we resolve `main` -> a concrete sha,
download THAT sha, and write it back into config.yaml (model.revision).
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, CONFIG_PATH  # noqa: E402


def main() -> None:
    cfg = load_config()
    repo_id = cfg["model"]["repo_id"]
    local_dir = cfg["model"]["local_dir"]

    # 1. Resolve the current commit sha of `main`.
    api = HfApi()
    info = api.model_info(repo_id, revision="main")
    sha = info.sha
    print(f"[resolve] {repo_id}@main -> {sha}")

    # 2. Download exactly that commit.
    path = snapshot_download(repo_id=repo_id, revision=sha, local_dir=local_dir)
    print(f"[download] -> {path}")

    # 3. Pin the sha in config.yaml (string replace to preserve comments/formatting).
    text = CONFIG_PATH.read_text()
    if "revision: FILL_AFTER_DOWNLOAD" in text:
        text = text.replace("revision: FILL_AFTER_DOWNLOAD", f"revision: {sha}")
        CONFIG_PATH.write_text(text)
        print(f"[pin] config.yaml model.revision = {sha}")
    else:
        # already pinned; warn if it differs from what we just downloaded
        current = load_config()["model"]["revision"]
        if current != sha:
            print(f"[WARN] config pins {current} but main is now {sha}; leaving config unchanged.")
        else:
            print(f"[pin] already pinned to {sha}")


if __name__ == "__main__":
    main()
