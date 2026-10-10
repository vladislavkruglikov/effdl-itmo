#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
export HF_HOME="${HF_HOME:-${TMPDIR:-/tmp}/cce-hf}"
cd ml-cross-entropy
env -u TRITON_PTXAS_PATH .venv/bin/python -m benchmark --test_data=mistral-nemo --methods=cce,torch_compile,baseline --kinds=loss-fw,loss-bw,loss-fw-bw --n_iteration=50 --n_rep=5 --output="../results/mistral_nemo_h100_5rep.csv"
