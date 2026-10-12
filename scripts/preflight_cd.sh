#!/usr/bin/env bash
set -euo pipefail
cd /home/toolkit/ELF
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PATH="$PWD/.venv-gpu/bin:$PATH"
PILOT_ROOT=${1:-artifacts/preflight}
mkdir -p "$PILOT_ROOT"
python -u -m t5_adaptation.calibrate --shared artifacts/shared --kind c4 \
  --output artifacts/shared/c4_normalization.json > "$PILOT_ROOT/calibration.log" 2>&1
exec torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  -m t5_adaptation.preflight --shared artifacts/shared --output "$PILOT_ROOT" \
  > "$PILOT_ROOT/pilot.log" 2>&1
