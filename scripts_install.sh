#!/usr/bin/env bash
# Install the full ML stack into .venv. Torch first (CUDA 12.4 wheels), then the rest.
set -e
cd /home/arun/Documents/mini-project
PY=.venv/bin/python

echo "=== [1/2] torch (cu124) ==="
$PY -m pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "=== [2/2] remaining stack (default index) ==="
$PY -m pip install \
  transformers peft datasets accelerate bitsandbytes trl \
  mergekit sqlglot sacrebleu jsonschema \
  pandas numpy matplotlib seaborn scipy \
  huggingface_hub pyyaml

echo "=== DONE. Versions: ==="
$PY - <<'EOF'
import torch, transformers, peft, datasets, trl
print("torch        ", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda device  ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("transformers ", transformers.__version__)
print("peft         ", peft.__version__)
print("datasets     ", datasets.__version__)
print("trl          ", trl.__version__)
EOF
