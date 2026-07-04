#!/usr/bin/env python3
"""
Fit the primary manuscript calibration:
Qwen2.5-7B-Instruct on RTX 3090, using vLLM GPU KV-cache usage (%) as the
memory-pressure variable and per-stream decode rate as the service-rate measure.

Expected output:
mu_max ≈ 53.61 tok/s
beta ≈ 0.0344 per KV-cache percentage point
R2 ≈ 0.996
RMSE ≈ 0.79 tok/s
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def exp_model(x, mu_max, beta):
    return mu_max * np.exp(-beta * x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/rtx_3090/qwen2.5_7b_instruct/processed/qwen_3090_kv_cache_summary.csv",
        help="CSV containing columns: kv_cache_usage_percent, decode_rate_tok_s",
    )
    parser.add_argument(
        "--outdir",
        default="data/rtx_3090/qwen2.5_7b_instruct/fit",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required = {"kv_cache_usage_percent", "decode_rate_tok_s"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = df["kv_cache_usage_percent"].to_numpy(dtype=float)
    y = df["decode_rate_tok_s"].to_numpy(dtype=float)

    popt, _ = curve_fit(exp_model, x, y, p0=[max(y), 0.03], maxfev=10000)
    mu_max, beta = popt
    yhat = exp_model(x, mu_max, beta)

    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y - yhat) ** 2))

    import os
    os.makedirs(args.outdir, exist_ok=True)

    summary_path = f"{args.outdir}/qwen_3090_kv_cache_fit_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Primary manuscript calibration: Qwen2.5-7B / RTX 3090\n")
        f.write("Load variable: vLLM GPU KV-cache usage (%)\n")
        f.write("Service variable: per-stream decode rate = 1000 / TPOT(ms)\n")
        f.write(f"mu_max={mu_max:.4f} tok/s\n")
        f.write(f"beta={beta:.6f} per KV-cache percentage point\n")
        f.write(f"R2={r2:.6f}\n")
        f.write(f"RMSE={rmse:.6f} tok/s\n")

    fit_values = df.copy()
    fit_values["fit_decode_rate_tok_s"] = yhat
    fit_values.to_csv(f"{args.outdir}/qwen_3090_kv_cache_fit_values.csv", index=False)

    plt.figure(figsize=(7.2, 4.6))
    plt.scatter(x, y, label="3-run mean")
    xs = np.linspace(min(x), max(x), 200)
    plt.plot(xs, exp_model(xs, mu_max, beta),
             label=rf"Fit: $\mu={mu_max:.2f}e^{{-{beta:.4f}x}}$")
    plt.xlabel("GPU KV-cache usage (%)")
    plt.ylabel("Per-stream decode rate (tokens/s)")
    plt.title("Qwen2.5-7B / RTX 3090: service degradation fit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/qwen_3090_kv_cache_fit.png", dpi=300)

    print(f"mu_max={mu_max:.4f} tok/s")
    print(f"beta={beta:.6f} per KV-cache percentage point")
    print(f"R2={r2:.6f}")
    print(f"RMSE={rmse:.6f} tok/s")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
