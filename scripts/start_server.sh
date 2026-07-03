#!/usr/bin/env bash
# start_server.sh — Start vLLM server for benchmarking
# Usage: ./start_server.sh [MODEL] [MAX_MODEL_LEN]

set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
MAX_LEN="${2:-4096}"

echo "Starting vLLM server..."
echo "  Model: $MODEL"
echo "  Max model length: $MAX_LEN"
echo "  Port: 8000"

mkdir -p logs

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization 0.85 \
    --served-model-name qwen7b \
    2>&1 | tee logs/vllm_server.log
