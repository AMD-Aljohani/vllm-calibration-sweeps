#!/usr/bin/env python3
"""
Plot all empirical fits (primary and robustness proxies)
and write the summary performance comparison CSV.
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def exp_model(x, mu_max, beta):
    return mu_max * np.exp(-beta * x)


def main():
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    summary_rows = []

    # 1. Qwen 3090 KV-Cache Fit
    qwen_3090_csv = "data/rtx_3090/qwen2.5_7b_instruct/processed/qwen_3090_kv_cache_summary.csv"
    if os.path.exists(qwen_3090_csv):
        df = pd.read_csv(qwen_3090_csv)
        x = df["kv_cache_usage_percent"].to_numpy(dtype=float)
        y = df["decode_rate_tok_s"].to_numpy(dtype=float)
        popt, _ = curve_fit(exp_model, x, y, p0=[53.61, 0.0344], maxfev=10000)
        mu_max, beta = popt
        yhat = exp_model(x, mu_max, beta)
        r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)
        rmse = np.sqrt(np.mean((y - yhat) ** 2))

        summary_rows.append({
            "model_type": "Qwen2.5-7B-Instruct",
            "hardware": "RTX 3090",
            "load_variable": "GPU KV-cache usage (%)",
            "mu_max": f"{mu_max:.4f}",
            "beta": f"{beta:.6f}",
            "R2": f"{r2:.6f}",
            "RMSE": f"{rmse:.6f}"
        })

        plt.figure(figsize=(7.2, 4.6))
        plt.scatter(x, y, color="tab:blue", label="3-run mean")
        xs = np.linspace(min(x), max(x), 200)
        plt.plot(xs, exp_model(xs, mu_max, beta), color="tab:red",
                 label=rf"Fit: $\mu={mu_max:.2f}e^{{-{beta:.4f}x}}$")
        plt.xlabel("GPU KV-cache usage (%)")
        plt.ylabel("Per-stream decode rate (tokens/s)")
        plt.title("Primary Calibration: Qwen2.5-7B / RTX 3090")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, "qwen_3090_kv_cache_fit.png"), dpi=300)
        plt.close()
        print("Generated qwen_3090_kv_cache_fit.png")

    # 2. Mistral 3090 Concurrency Fit
    mistral_3090_csv = "data/rtx_3090/mistral_7b_instruct_v0.3/processed/mistral_3run_summary.csv"
    if os.path.exists(mistral_3090_csv):
        df = pd.read_csv(mistral_3090_csv)
        x = df["concurrency"].to_numpy(dtype=float)
        y = df["decode_rate_tok_s"].to_numpy(dtype=float)
        popt, _ = curve_fit(exp_model, x, y, p0=[54.33, 0.0514], maxfev=10000)
        mu_max, beta = popt
        yhat = exp_model(x, mu_max, beta)
        r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)
        rmse = np.sqrt(np.mean((y - yhat) ** 2))

        summary_rows.append({
            "model_type": "Mistral-7B-Instruct-v0.3",
            "hardware": "RTX 3090",
            "load_variable": "Concurrency",
            "mu_max": f"{mu_max:.4f}",
            "beta": f"{beta:.6f}",
            "R2": f"{r2:.6f}",
            "RMSE": f"{rmse:.6f}"
        })

        plt.figure(figsize=(7.2, 4.6))
        plt.scatter(x, y, color="tab:green", label="3-run mean")
        xs = np.linspace(min(x), max(x), 200)
        plt.plot(xs, exp_model(xs, mu_max, beta), color="tab:red",
                 label=rf"Fit: $\mu={mu_max:.2f}e^{{-{beta:.4f}C}}$")
        plt.xlabel("Concurrency (C)")
        plt.ylabel("Per-stream decode rate (tokens/s)")
        plt.title("Robustness Proxy: Mistral-7B / RTX 3090")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, "mistral_3090_fit.png"), dpi=300)
        plt.close()
        print("Generated mistral_3090_fit.png")

    # 3. Qwen 4090 Concurrency Fit
    qwen_4090_csv = "data/rtx_4090/qwen2.5_7b_instruct/processed/qwen_4090_summary.csv"
    if os.path.exists(qwen_4090_csv):
        df = pd.read_csv(qwen_4090_csv)
        x = df["concurrency"].to_numpy(dtype=float)
        y = df["decode_rate_tok_s"].to_numpy(dtype=float)
        popt, _ = curve_fit(exp_model, x, y, p0=[64.00, 0.0280], maxfev=10000)
        mu_max, beta = popt
        yhat = exp_model(x, mu_max, beta)
        r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)
        rmse = np.sqrt(np.mean((y - yhat) ** 2))

        summary_rows.append({
            "model_type": "Qwen2.5-7B-Instruct",
            "hardware": "RTX 4090",
            "load_variable": "Concurrency",
            "mu_max": f"{mu_max:.4f}",
            "beta": f"{beta:.6f}",
            "R2": f"{r2:.6f}",
            "RMSE": f"{rmse:.6f}"
        })

        plt.figure(figsize=(7.2, 4.6))
        plt.scatter(x, y, color="tab:orange", label="3-run mean")
        xs = np.linspace(min(x), max(x), 200)
        plt.plot(xs, exp_model(xs, mu_max, beta), color="tab:red",
                 label=rf"Fit: $\mu={mu_max:.2f}e^{{-{beta:.4f}C}}$")
        plt.xlabel("Concurrency (C)")
        plt.ylabel("Per-stream decode rate (tokens/s)")
        plt.title("Robustness Proxy: Qwen2.5-7B / RTX 4090")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, "qwen_4090_concurrency_fit.png"), dpi=300)
        plt.close()
        print("Generated qwen_4090_concurrency_fit.png")

    # Write summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(outputs_dir, "model_hardware_robustness_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Wrote performance summary to {summary_csv_path}")


if __name__ == "__main__":
    main()
