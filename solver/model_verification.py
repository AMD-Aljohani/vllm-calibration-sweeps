#!/usr/bin/env python3
"""
Verify that the generated model parameters and fits match the manuscript specifications.
"""

import os
import pandas as pd


def main():
    summary_path = "outputs/model_hardware_robustness_summary.csv"
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found. Please run scripts/plot_all_fits.py first.")
        exit(1)

    df = pd.read_csv(summary_path)

    print("==================================================")
    # Target values/bounds for validation
    valid = True

    # 1. Qwen 3090 KV Cache fit
    qwen_3090 = df[(df["model_type"] == "Qwen2.5-7B-Instruct") & (df["hardware"] == "RTX 3090")]
    if qwen_3090.empty:
        print("[FAIL] Qwen2.5-7B-Instruct / RTX 3090 entry missing.")
        valid = False
    else:
        mu_max = float(qwen_3090.iloc[0]["mu_max"])
        beta = float(qwen_3090.iloc[0]["beta"])
        r2 = float(qwen_3090.iloc[0]["R2"])
        rmse = float(qwen_3090.iloc[0]["RMSE"])

        print("Qwen2.5-7B / RTX 3090 (KV-cache usage % fit):")
        print(f"  mu_max = {mu_max:.4f} (Expected ≈ 53.61)")
        print(f"  beta   = {beta:.4f} (Expected ≈ 0.0344)")
        print(f"  R2     = {r2:.4f} (Expected >= 0.99)")
        print(f"  RMSE   = {rmse:.4f} (Expected ≈ 0.79)")

        # Verify bounds
        if not (53.0 <= mu_max <= 54.0):
            print("  [FAIL] mu_max out of expected range.")
            valid = False
        elif not (0.033 <= beta <= 0.036):
            print("  [FAIL] beta out of expected range.")
            valid = False
        elif r2 < 0.99:
            print("  [FAIL] R2 lower than expected (>= 0.99).")
            valid = False
        elif not (0.7 <= rmse <= 0.9):
            print("  [FAIL] RMSE out of expected range.")
            valid = False
        else:
            print("  [PASS] All parameters match manuscript specifications.")

    print("--------------------------------------------------")

    # 2. Mistral 3090 Concurrency fit
    mistral_3090 = df[(df["model_type"] == "Mistral-7B-Instruct-v0.3") & (df["hardware"] == "RTX 3090")]
    if mistral_3090.empty:
        print("[FAIL] Mistral-7B-Instruct-v0.3 / RTX 3090 entry missing.")
        valid = False
    else:
        r2 = float(mistral_3090.iloc[0]["R2"])
        print("Mistral-7B / RTX 3090 (concurrency fit):")
        print(f"  R2 = {r2:.4f} (Expected >= 0.97)")
        if r2 < 0.97:
            print("  [FAIL] R2 lower than expected (>= 0.97).")
            valid = False
        else:
            print("  [PASS] R2 matches manuscript specifications.")

    print("--------------------------------------------------")

    # 3. Qwen 4090 Concurrency fit
    qwen_4090 = df[(df["model_type"] == "Qwen2.5-7B-Instruct") & (df["hardware"] == "RTX 4090")]
    if qwen_4090.empty:
        print("[FAIL] Qwen2.5-7B-Instruct / RTX 4090 entry missing.")
        valid = False
    else:
        r2 = float(qwen_4090.iloc[0]["R2"])
        print("Qwen2.5-7B / RTX 4090 (concurrency fit):")
        print(f"  R2 = {r2:.4f} (Expected >= 0.99)")
        if r2 < 0.99:
            print("  [FAIL] R2 lower than expected (>= 0.99).")
            valid = False
        else:
            print("  [PASS] R2 matches manuscript specifications.")

    print("==================================================")
    if valid:
        print("Overall verification result: SUCCESS")
        exit(0)
    else:
        print("Overall verification result: FAILED")
        exit(1)


if __name__ == "__main__":
    main()
