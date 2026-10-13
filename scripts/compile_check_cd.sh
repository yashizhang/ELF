#!/usr/bin/env bash
set -euo pipefail
cd /home/toolkit/ELF
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export CC="$PWD/artifacts/compiler/bin/x86_64-conda-linux-gnu-cc"
export CXX="$PWD/artifacts/compiler/bin/x86_64-conda-linux-gnu-c++"
export TORCHINDUCTOR_CACHE_DIR="$PWD/artifacts/compile-check/inductor"
mkdir -p artifacts/compile-check
exec .venv-gpu/bin/python -u -m t5_adaptation.compile_check \
  --shared artifacts/shared --output artifacts/compile-check \
  > artifacts/compile-check/run.log 2>&1
