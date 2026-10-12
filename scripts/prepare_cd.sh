#!/usr/bin/env bash
set -euo pipefail
cd /home/toolkit/ELF
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=28
export TOKENIZERS_PARALLELISM=true
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
mkdir -p artifacts/shared
exec .venv/bin/python -u -m t5_adaptation.prepare \
  --root /home/toolkit/ELF/artifacts/shared --stage all \
  >> artifacts/shared/preparation.log 2>&1
