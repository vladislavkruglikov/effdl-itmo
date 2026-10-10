#!/usr/bin/env bash
set -euo pipefail
cd ml-cross-entropy
env -u TRITON_PTXAS_PATH .venv/bin/python -m pytest --disable-warnings --maxfail=1 "$@"
