#!/usr/bin/env bash
set -euo pipefail
test "${GUIDANCE_ACCOUNT:?}" = snow.research.er
STUDY_ARM=$1
STUDY_PHASE=$2
STUDY_ROOT=$3
mkdir -p "$STUDY_ROOT/$STUDY_ARM"
exec /home/toolkit/ELF/.venv-gpu/bin/python -u -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=8 -m guidance_sweep.run \
  --arm "$STUDY_ARM" --phase "$STUDY_PHASE" --root "$STUDY_ROOT" \
  > "$STUDY_ROOT/$STUDY_ARM/$STUDY_PHASE-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
