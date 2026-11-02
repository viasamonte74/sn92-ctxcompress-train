#!/usr/bin/env bash
# Serve a model the way the official engine image does for the SLM pool:
# batch invariant mode, seed 0, the arena's context (122,880 in + 3,072 out), served as "slm".
#   scripts/serve_vllm.sh MODEL_DIR [PORT] [extra vllm args, e.g. --quantization gptq_marlin]
set -euo pipefail
MODEL=${1:?model folder}
PORT=${2:-8000}
shift $(( $# >= 2 ? 2 : $# ))
# vLLM 0.32 refuses batch invariant mode for Granite 4.0 H (MAMBA2_ATTN backend), so local evals run
# without it by default; VLLM_BATCH_INVARIANT=1 restores the official P1 setting once vLLM supports it.
export VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-0} VLLM_NO_USAGE_STATS=1
# the project venv's vllm, so this works without activating it (pm2, cron, a plain shell)
VENV_BIN="$(cd "$(dirname "$0")/.." && pwd)/.venv/bin"
[[ -x "$VENV_BIN/vllm" ]] && export PATH="$VENV_BIN:$PATH"
exec vllm serve "$MODEL" \
  --served-model-name slm \
  --host 127.0.0.1 --port "$PORT" \
  --max-model-len 125952 \
  --seed 0 \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
  "$@"
