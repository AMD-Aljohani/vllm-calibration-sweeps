#!/usr/bin/env python3
"""
Fit a concurrency-proxy exponential degradation model:
mu(C) = mu_max * exp(-beta * C)
where C is the request concurrency. Used for the secondary robustness proxies.
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def exp_model(C, mu_max, beta):
    return mu_max * np.exp(-beta * C)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to processed CSV")
    parser.add_argument("--out", required=True, help="Output path for summary TXT")
    parser.add_argument("--label", required=True, help="Label for this fit")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required = {"concurrency", "decode_rate_tok_s"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    C = df["concurrency"].to_numpy(dtype=float)
    y = df["decode_rate_tok_s"].to_numpy(dtype=float)

    popt, _ = curve_fit(exp_model, C, y, p0=[max(y), 0.03], maxfev=10000)
    mu_max, beta = popt
    yhat = exp_model(C, mu_max, beta)

    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y - yhat) ** 2))

    import os
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w") as f:
        f.write(f"Label: {args.label}\n")
        f.write(f"mu_max={mu_max:.4f} tok/s\n")
        f.write(f"beta={beta:.6f} per concurrency point\n")
        f.write(f"R2={r2:.6f}\n")
        f.write(f"RMSE={rmse:.6f} tok/s\n")

    print(f"[{args.label}]")
    print(f"mu_max={mu_max:.4f} tok/s")
    print(f"beta={beta:.6f} per concurrency point")
    print(f"R2={r2:.6f}")
    print(f"RMSE={rmse:.6f} tok/s")
    print(f"Wrote summary to {args.out}")


if __name__ == "__main__":
    main()
