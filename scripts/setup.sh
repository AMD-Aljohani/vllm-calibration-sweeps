#!/usr/bin/env bash
# setup.sh — Install all dependencies on RunPod
set -euo pipefail

echo "═══ System packages ═══"
apt update
apt install -y unzip git htop tmux nvtop

echo ""
echo "═══ Python packages ═══"
python3 -m pip install -U pip
python3 -m pip install -U vllm pandas numpy scipy matplotlib requests

echo ""
echo "═══ GPU check ═══"
nvidia-smi
python3 - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

echo ""
echo "═══ vLLM version ═══"
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"

echo ""
echo "Setup complete!"
