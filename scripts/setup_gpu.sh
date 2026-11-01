#!/usr/bin/env bash
# Set up a rented GPU instance (vast.ai or similar) for training and evaluation.
# Run from the ctx-compress-slm folder; finds microtensor-arenas/-subnet next to it or in MT_REPOS:
#   bash scripts/setup_gpu.sh            # training + vLLM + probe
#   bash scripts/setup_gpu.sh --quant    # also a separate venv for GPTQModel (experimental)
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf}
VLLM_VERSION=${VLLM_VERSION:-0.32.0}   # the engine image the arena pins: vllm/vllm-openai:v0.32.0

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h . | tail -1
# The microtensor repositories: MT_REPOS, else next to this folder, else three levels up (the zip's
# layout unpacked inside an existing microtensor/ checkout), else cloned next to this folder.
if [ -z "${MT_REPOS:-}" ]; then
  for candidate in .. ../../..; do
    if [ -d "$candidate/microtensor-subnet" ] && [ -d "$candidate/microtensor-arenas" ]; then MT_REPOS=$candidate; break; fi
  done
fi
MT_REPOS=$(realpath "${MT_REPOS:-..}")
echo "microtensor repositories in $MT_REPOS"
for repo in microtensor-arenas microtensor-subnet microtensor-miner-template; do
  [ -d "$MT_REPOS/$repo" ] || git clone -q "https://github.com/microtensor-io/$repo" "$MT_REPOS/$repo"
done

command -v uv >/dev/null || python3 -m pip install -q uv
uv venv -q --python 3.12 .venv
. .venv/bin/activate
uv pip install -q "vllm==${VLLM_VERSION}"
uv pip install -q -e . pytest bitsandbytes   # bitsandbytes: 8 bit AdamW moments for --method full
uv pip install -q --no-deps -e "$MT_REPOS/microtensor-subnet" -e "$MT_REPOS/microtensor-arenas"
# Mamba-2 training kernels. Without mamba_ssm and causal_conv1d transformers falls back to its
# reference PyTorch scan, which runs out of memory on a single 16k token example even on 48 GB.
# Hub "kernels" are only used when asked for, so they do not count. Both packages build from source
# against the local nvcc for this GPU's architecture (~10 min with 16 cores); --no-deps keeps vLLM's
# pins (mamba-ssm would downgrade tilelang and apache-tvm-ffi).
if ! python -c "import mamba_ssm, causal_conv1d" 2>/dev/null; then
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=${CUDA_HOME:-/usr/local/cuda}/bin:$PATH MAX_JOBS=${MAX_JOBS:-$(nproc)}
  export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$(python -c "import torch; print('%d.%d' % torch.cuda.get_device_capability())")}
  uv pip install -q ninja packaging setuptools wheel einops
  CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE \
    uv pip install -q --no-build-isolation --no-cache --no-deps causal-conv1d==1.7.0 mamba-ssm==2.3.2.post1
fi
python -c "import mamba_ssm, causal_conv1d; from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined; print('mamba kernels ok')"

if [ "${1:-}" = "--quant" ]; then
  uv venv -q --python 3.12 .venv-quant
  VIRTUAL_ENV=.venv-quant uv pip install -q gptqmodel -e . && VIRTUAL_ENV=.venv-quant uv pip install -q --no-deps -e "$MT_REPOS/microtensor-subnet" -e "$MT_REPOS/microtensor-arenas"
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
