#!/usr/bin/env bash
# Set up a rented GPU instance (vast.ai or similar) for training and evaluation.
# Run from the ctx-compress-slm folder, with microtensor-arenas and microtensor-subnet next to it:
#   bash scripts/setup_gpu.sh            # training + vLLM + probe
#   bash scripts/setup_gpu.sh --quant    # also a separate venv for GPTQModel (experimental)
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf}
VLLM_VERSION=${VLLM_VERSION:-0.32.0}   # the engine image the arena pins: vllm/vllm-openai:v0.32.0

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h . | tail -1
[ -d ../microtensor-arenas ] || git clone -q https://github.com/microtensor-io/microtensor-arenas ../microtensor-arenas
[ -d ../microtensor-subnet ] || git clone -q https://github.com/microtensor-io/microtensor-subnet ../microtensor-subnet
[ -d ../microtensor-miner-template ] || git clone -q https://github.com/microtensor-io/microtensor-miner-template ../microtensor-miner-template

command -v uv >/dev/null || python3 -m pip install -q uv
uv venv -q --python 3.12 .venv
. .venv/bin/activate
uv pip install -q "vllm==${VLLM_VERSION}"
uv pip install -q -e . pytest bitsandbytes   # bitsandbytes: 8 bit AdamW moments for --method full
uv pip install -q --no-deps -e ../microtensor-subnet -e ../microtensor-arenas
# Mamba-2 training kernels: transformers takes them from the hub ("kernels") or from mamba_ssm.
uv pip install -q kernels || true
python - <<'PY' || uv pip install -q --no-build-isolation causal-conv1d mamba-ssm
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("mamba_ssm") or importlib.util.find_spec("kernels") else 1)
PY

if [ "${1:-}" = "--quant" ]; then
  uv venv -q --python 3.12 .venv-quant
  VIRTUAL_ENV=.venv-quant uv pip install -q gptqmodel -e . && VIRTUAL_ENV=.venv-quant uv pip install -q --no-deps -e ../microtensor-subnet -e ../microtensor-arenas
fi

python - <<'PY'
import torch, transformers, vllm
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__, "vllm", vllm.__version__)
from huggingface_hub import snapshot_download
snapshot_download("ibm-granite/granite-4.0-h-micro", revision="d5f01a3ea75f088947be3aae039f4ad52837dfde")
snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
print("models cached under", __import__("os").environ.get("HF_HOME"))
PY
python -m pytest -q tests
