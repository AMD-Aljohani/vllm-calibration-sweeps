#!/usr/bin/env python3
"""
aggregate_results.py
====================
Aggregates vLLM benchmark JSON results and GPU nvidia-smi CSV logs
into a single calibration_summary.csv for degradation model fitting.

Usage:
    python aggregate_results.py \
        --results-dir results \
        --gpu-log-dir gpu_logs \
        --out calibration_summary.csv
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd


def parse_vllm_result(filepath: str) -> dict:
    """Parse a single vLLM benchmark result JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    # Extract concurrency from filename: result_concurrency_N.json
    basename = os.path.basename(filepath)
    match = re.search(r"concurrency_(\d+)", basename)
    if not match:
        raise ValueError(f"Cannot extract concurrency from filename: {basename}")
    concurrency = int(match.group(1))

    # vLLM benchmark_serving.py outputs vary by version.
    # We handle multiple known formats.
    result = {"concurrency": concurrency}

    # Try to extract key metrics
    # Mean TPOT (time per output token) in ms
    for key in [
        "mean_tpot_ms",
        "avg_tpot_ms",
        "mean_inter_token_latency_ms",
        "avg_per_output_token_latency_ms",
    ]:
        if key in data:
            result["tpot_ms"] = float(data[key])
            break

    # Mean TTFT (time to first token) in ms
    for key in [
        "mean_ttft_ms",
        "avg_ttft_ms",
        "mean_time_to_first_token_ms",
        "avg_time_to_first_token_ms",
    ]:
        if key in data:
            result["ttft_ms"] = float(data[key])
            break

    # Throughput (tokens/sec)
    for key in [
        "output_throughput",
        "total_output_throughput",
        "completed_request_throughput",
        "request_throughput",
    ]:
        if key in data:
            result["throughput_tok_s"] = float(data[key])
            break

    # Request throughput (requests/sec)
    for key in ["request_throughput", "completed_request_rate"]:
        if key in data and "request_throughput" not in result:
            result["request_throughput"] = float(data[key])

    # Median / P50 TPOT
    for key in ["median_tpot_ms", "p50_tpot_ms", "median_inter_token_latency_ms"]:
        if key in data:
            result["tpot_p50_ms"] = float(data[key])
            break

    # P99 TPOT
    for key in ["p99_tpot_ms", "p99_inter_token_latency_ms"]:
        if key in data:
            result["tpot_p99_ms"] = float(data[key])
            break

    # P99 TTFT
    for key in ["p99_ttft_ms", "p99_time_to_first_token_ms"]:
        if key in data:
            result["ttft_p99_ms"] = float(data[key])
            break

    # Total duration
    for key in ["total_time", "duration", "elapsed_time"]:
        if key in data:
            result["total_time_s"] = float(data[key])
            break

    # Number of completed requests
    for key in ["completed", "num_completed", "num_prompts_completed"]:
        if key in data:
            result["num_completed"] = int(data[key])
            break

    # Mean input/output lengths
    for key in ["mean_input_len", "avg_input_len", "mean_input_tokens"]:
        if key in data:
            result["mean_input_len"] = float(data[key])
            break

    for key in ["mean_output_len", "avg_output_len", "mean_output_tokens"]:
        if key in data:
            result["mean_output_len"] = float(data[key])
            break

    # E2E latency
    for key in [
        "mean_e2e_latency_ms",
        "avg_latency_ms",
        "mean_request_latency_ms",
        "avg_e2e_latency_ms",
    ]:
        if key in data:
            result["e2e_latency_ms"] = float(data[key])
            break

    return result


def parse_gpu_log(filepath: str) -> dict:
    """
    Parse an nvidia-smi CSV log file.
    Returns aggregated GPU metrics (mean, max).
    """
    try:
        df = pd.read_csv(filepath, skipinitialspace=True)
    except Exception as e:
        print(f"  Warning: Could not parse GPU log {filepath}: {e}")
        return {}

    # Clean column names (nvidia-smi adds spaces and units)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    result = {}

    # GPU memory used (MiB)
    for col in df.columns:
        if "memory.used" in col or "memory_used" in col:
            # Strip ' MiB' suffix if present
            vals = df[col].astype(str).str.replace(r"[^\d.]", "", regex=True)
            vals = pd.to_numeric(vals, errors="coerce").dropna()
            if len(vals) > 0:
                result["gpu_mem_used_mib_mean"] = round(vals.mean(), 1)
                result["gpu_mem_used_mib_max"] = round(vals.max(), 1)
            break

    # GPU memory total (MiB)
    for col in df.columns:
        if "memory.total" in col or "memory_total" in col:
            vals = df[col].astype(str).str.replace(r"[^\d.]", "", regex=True)
            vals = pd.to_numeric(vals, errors="coerce").dropna()
            if len(vals) > 0:
                result["gpu_mem_total_mib"] = round(vals.iloc[0], 1)
            break

    # GPU utilization (%)
    for col in df.columns:
        if "utilization.gpu" in col or "utilization_gpu" in col:
            vals = df[col].astype(str).str.replace(r"[^\d.]", "", regex=True)
            vals = pd.to_numeric(vals, errors="coerce").dropna()
            if len(vals) > 0:
                result["gpu_util_pct_mean"] = round(vals.mean(), 1)
                result["gpu_util_pct_max"] = round(vals.max(), 1)
            break

    # Power draw (W)
    for col in df.columns:
        if "power.draw" in col or "power_draw" in col:
            vals = df[col].astype(str).str.replace(r"[^\d.]", "", regex=True)
            vals = pd.to_numeric(vals, errors="coerce").dropna()
            if len(vals) > 0:
                result["power_w_mean"] = round(vals.mean(), 1)
                result["power_w_max"] = round(vals.max(), 1)
            break

    # Temperature (°C)
    for col in df.columns:
        if "temperature" in col:
            vals = df[col].astype(str).str.replace(r"[^\d.]", "", regex=True)
            vals = pd.to_numeric(vals, errors="coerce").dropna()
            if len(vals) > 0:
                result["temp_c_mean"] = round(vals.mean(), 1)
                result["temp_c_max"] = round(vals.max(), 1)
            break

    # Compute memory pressure x = mem_used / mem_total
    if "gpu_mem_used_mib_mean" in result and "gpu_mem_total_mib" in result:
        mem_total = result["gpu_mem_total_mib"]
        if mem_total > 0:
            result["mem_pressure_x_mean"] = round(
                result["gpu_mem_used_mib_mean"] / mem_total, 4
            )
            result["mem_pressure_x_max"] = round(
                result["gpu_mem_used_mib_max"] / mem_total, 4
            )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate vLLM benchmark results and GPU logs"
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing result_concurrency_*.json files",
    )
    parser.add_argument(
        "--gpu-log-dir",
        required=True,
        help="Directory containing gpu_concurrency_*.csv files",
    )
    parser.add_argument(
        "--out", default="calibration_summary.csv", help="Output CSV filename"
    )
    args = parser.parse_args()

    # Parse all result JSON files
    result_files = sorted(
        [
            f
            for f in os.listdir(args.results_dir)
            if f.startswith("result_concurrency_") and f.endswith(".json")
        ]
    )

    if not result_files:
        print(f"ERROR: No result_concurrency_*.json files found in {args.results_dir}")
        sys.exit(1)

    print(f"Found {len(result_files)} result files")

    rows = []
    for rf in result_files:
        filepath = os.path.join(args.results_dir, rf)
        print(f"  Parsing: {rf}")
        try:
            row = parse_vllm_result(filepath)
        except Exception as e:
            print(f"  ERROR parsing {rf}: {e}")
            continue

        # Find matching GPU log
        conc = row["concurrency"]
        gpu_log = os.path.join(args.gpu_log_dir, f"gpu_concurrency_{conc}.csv")
        if os.path.exists(gpu_log):
            print(f"  Parsing GPU log: gpu_concurrency_{conc}.csv")
            gpu_data = parse_gpu_log(gpu_log)
            row.update(gpu_data)
        else:
            print(f"  Warning: No GPU log for concurrency {conc}")

        # Compute derived metrics
        if "tpot_ms" in row and row["tpot_ms"] > 0:
            # Decode service rate: μ_decode = 1000 / TPOT(ms)  [tokens/sec]
            row["decode_rate_tok_s"] = round(1000.0 / row["tpot_ms"], 4)

        rows.append(row)

    if not rows:
        print("ERROR: No valid results parsed")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df = df.sort_values("concurrency").reset_index(drop=True)

    # Reorder columns for readability
    priority_cols = [
        "concurrency",
        "tpot_ms",
        "decode_rate_tok_s",
        "ttft_ms",
        "throughput_tok_s",
        "e2e_latency_ms",
        "gpu_mem_used_mib_mean",
        "gpu_mem_used_mib_max",
        "gpu_mem_total_mib",
        "mem_pressure_x_mean",
        "mem_pressure_x_max",
        "gpu_util_pct_mean",
        "power_w_mean",
        "temp_c_mean",
    ]
    existing_priority = [c for c in priority_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in existing_priority]
    df = df[existing_priority + remaining]

    df.to_csv(args.out, index=False, float_format="%.4f")
    print(f"\nSaved: {args.out}")
    print(f"Shape: {df.shape}")
    print(f"\nSummary:\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
