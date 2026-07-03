#!/usr/bin/env bash
# run_sweep.sh — Run the concurrency sweep benchmark
# Usage: ./run_sweep.sh [MODEL_SERVED]

set -euo pipefail

MODEL_SERVED="${1:-qwen7b}"
CONCURRENCY_LEVELS="1 2 4 8 16 32"
NUM_PROMPTS=256
INPUT_LEN=1024
OUTPUT_LEN=128

echo "═══════════════════════════════════════════════════════"
echo "  vLLM Concurrency Sweep Benchmark"
echo "  Model: $MODEL_SERVED"
echo "  Concurrency: $CONCURRENCY_LEVELS"
echo "  Prompts: $NUM_PROMPTS  Input: $INPUT_LEN  Output: $OUTPUT_LEN"
echo "═══════════════════════════════════════════════════════"

mkdir -p results gpu_logs logs

for C in $CONCURRENCY_LEVELS; do
    echo ""
    echo "──── Concurrency $C ────"
    echo "Started at: $(date)"

    # Start GPU monitoring in background
    nvidia-smi \
        --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
        --format=csv \
        -l 1 > "gpu_logs/gpu_concurrency_${C}.csv" &
    SMI_PID=$!

    # Run benchmark
    vllm bench serve \
        --backend vllm \
        --host 127.0.0.1 \
        --port 8000 \
        --model "$MODEL_SERVED" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$C" \
        --save-result \
        --result-filename "results/result_concurrency_${C}.json" \
        2>&1 | tee "logs/bench_concurrency_${C}.log"

    # Stop GPU monitoring
    kill $SMI_PID 2>/dev/null || true
    wait $SMI_PID 2>/dev/null || true

    echo "Completed at: $(date)"
    echo "Cooldown 20s..."
    sleep 20
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Sweep complete!"
echo "═══════════════════════════════════════════════════════"
