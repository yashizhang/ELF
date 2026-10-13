#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-gpu}
if [[ "$MODE" == gpu ]]; then
  ENV_PATH=.venv-gpu
  TORCH_BUILD=2.5.1+cu124
  TORCH_INDEX=https://download.pytorch.org/whl/cu124
  DEPENDENCY_LOCK=requirements-adaptation-gpu.lock
elif [[ "$MODE" == cpu ]]; then
  ENV_PATH=.venv
  TORCH_BUILD=2.5.1+cpu
  TORCH_INDEX=https://download.pytorch.org/whl/cpu
  DEPENDENCY_LOCK=requirements-adaptation.lock
else
  echo 'usage: setup_adaptation_env.sh [gpu|cpu]' >&2
  exit 2
fi
UV=${UV:-/home/toolkit/.local/bin/uv}
if [[ ! -x "$ENV_PATH/bin/python" ]]; then
  "$UV" venv --python 3.11.15 "$ENV_PATH"
fi
"$UV" --no-config pip install --python "$ENV_PATH/bin/python" \
  --exclude-newer 2026-10-13 --index "$TORCH_INDEX" --index-strategy unsafe-best-match \
  "torch==$TORCH_BUILD" 'sympy==1.13.1'
"$UV" --no-config pip install --python "$ENV_PATH/bin/python" \
  --exclude-newer 2026-10-13 -r "$DEPENDENCY_LOCK"
"$UV" --no-config pip install --python "$ENV_PATH/bin/python" --no-deps \
  -e ./third_party/transformers -e ./third_party/lejepa
"$UV" pip check --python "$ENV_PATH/bin/python"
if [[ "$MODE" == gpu && ! -x artifacts/compiler/bin/x86_64-conda-linux-gnu-c++ ]]; then
  micromamba create -y -p "$PWD/artifacts/compiler" --file compiler-linux-64.lock
fi
"$ENV_PATH/bin/python" -c 'import sys, torch, transformers; print(sys.version); print(torch.__version__, torch.version.cuda, transformers.__version__)'
