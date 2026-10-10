#!/usr/bin/env bash
set -euo pipefail
if [[ ! -d ml-cross-entropy ]]; then
  git clone https://github.com/apple/ml-cross-entropy.git
fi
cd ml-cross-entropy
[[ -z "$(git status --porcelain)" ]] || { echo 'Есть изменения в исходниках; остановка.' >&2; exit 1; }
git checkout --detach 24fbe4b5dab9a6c250a014573613c1890190536c
if [[ ! -d .venv ]]; then uv venv --python python3.12 .venv; fi
if grep -q 'include-system-site-packages = true' .venv/pyvenv.cfg; then
  echo 'Нужна изолированная .venv без системных пакетов.' >&2
  exit 1
fi
uv pip sync --python .venv/bin/python --index-url https://pypi.org/simple ../requirements-repro.lock
uv pip install --python .venv/bin/python --no-deps -e .
uv pip check --python .venv/bin/python
.venv/bin/python -c 'import torch,triton,transformers,flash_attn; from transformers.trainer import EvalPrediction; print("torch",torch.__version__,"CUDA",torch.version.cuda,"triton",triton.__version__,"transformers",transformers.__version__,"flash_attn",flash_attn.__version__); assert torch.cuda.is_available()'
git diff --exit-code
