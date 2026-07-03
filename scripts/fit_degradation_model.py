#!/usr/bin/env python3
"""
fit_degradation_model.py - Fits μ(x) = μ_max * exp(-β * x) to calibration data.
Usage: python fit_degradation_model.py --csv calibration_summary.csv --out-prefix fit_output
"""
import argparse, sys, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def exp_decay(x, mu_max, beta):
    return mu_max * np.exp(-beta * x)

def r_squared(y, yp):
    ss_res = np.sum((y - yp)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res/ss_tot if ss_tot else 0.0

def rmse(y, yp):
    return np.sqrt(np.mean((y - yp)**2))

def do_fit(x, y, label):
    if len(x) < 2:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(exp_decay, x, y, p0=[y.max()*1.1, 0.1],
                                   maxfev=10000, bounds=([0,0],[np.inf,np.inf]))
        perr = np.sqrt(np.diag(pcov))
        yp = exp_decay(x, *popt)
        r = {"label": label, "mu_max": popt[0], "mu_max_se": perr[0],
             "beta": popt[1], "beta_se": perr[1],
             "r2": r_squared(y, yp), "rmse_val": rmse(y, yp),
             "n": len(x), "x": x, "y": y, "yp": yp}
        print(f"  {label}: mu_max={r['mu_max']:.4f}±{r['mu_max_se']:.4f}, "
              f"beta={r['beta']:.6f}±{r['beta_se']:.6f}, R²={r['r2']:.6f}, RMSE={r['rmse_val']:.4f}")
        return r
    except Exception as e:
        print(f"  {label}: fit failed: {e}"); return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out-prefix", default="fit_output")
    a = p.parse_args()
    df = pd.read_csv(a.csv)
    print(f"Loaded {len(df)} rows")
    fits = []

    # Fit 1: decode rate vs concurrency
    if {"concurrency","decode_rate_tok_s"} <= set(df.columns):
        m = df["decode_rate_tok_s"].notna() & df["concurrency"].notna()
        if m.sum() >= 2:
            r = do_fit(df.loc[m,"concurrency"].values.astype(float),
                       df.loc[m,"decode_rate_tok_s"].values.astype(float),
                       "decode_rate_vs_concurrency")
            if r: fits.append(r)

    # Fit 2: decode rate vs memory pressure
    if {"mem_pressure_x_mean","decode_rate_tok_s"} <= set(df.columns):
        m = df["decode_rate_tok_s"].notna() & df["mem_pressure_x_mean"].notna()
        if m.sum() >= 2:
            r = do_fit(df.loc[m,"mem_pressure_x_mean"].values.astype(float),
                       df.loc[m,"decode_rate_tok_s"].values.astype(float),
                       "decode_rate_vs_mem_pressure")
            if r: fits.append(r)

    if not fits:
        print("No successful fits"); sys.exit(1)

    # Save parameters
    with open(f"{a.out_prefix}_parameters.txt","w") as f:
        f.write("="*70+"\nExponential Degradation Model: μ(x) = μ_max × exp(−β × x)\n"+"="*70+"\n\n")
        for fi in fits:
            f.write(f"Model: {fi['label']}\n"+"-"*50+"\n")
            f.write(f"  μ_max  = {fi['mu_max']:.6f} ± {fi['mu_max_se']:.6f}\n")
            f.write(f"  β      = {fi['beta']:.8f} ± {fi['beta_se']:.8f}\n")
            f.write(f"  R²     = {fi['r2']:.6f}\n")
            f.write(f"  RMSE   = {fi['rmse_val']:.6f}\n")
            f.write(f"  N      = {fi['n']}\n\n")
        f.write("\n"+"="*70+"\nRaw Data\n"+"="*70+"\n\n"+df.to_string(index=False)+"\n")
    print(f"Saved: {a.out_prefix}_parameters.txt")

    # Save fit CSV
    rows = [{"model":fi["label"],"mu_max":round(fi["mu_max"],6),"mu_max_se":round(fi["mu_max_se"],6),
             "beta":round(fi["beta"],8),"beta_se":round(fi["beta_se"],8),
             "r_squared":round(fi["r2"],6),"rmse":round(fi["rmse_val"],6),"n":fi["n"]} for fi in fits]
    pd.DataFrame(rows).to_csv(f"{a.out_prefix}_fit.csv", index=False)
    print(f"Saved: {a.out_prefix}_fit.csv")

    # Plot
    fig, axes = plt.subplots(1, len(fits), figsize=(7*len(fits),6), squeeze=False)
    fig.suptitle("μ(x) = μ_max · exp(−β · x)", fontsize=14, fontweight="bold")
    for i, fi in enumerate(fits):
        ax = axes[0,i]
        si = np.argsort(fi["x"])
        xd = np.linspace(fi["x"].min(), fi["x"].max()*1.1, 200)
        ax.scatter(fi["x"][si], fi["y"][si], c="#2196F3", s=80, zorder=5, label="Measured", edgecolors="w")
        ax.plot(xd, exp_decay(xd, fi["mu_max"], fi["beta"]), "r--", lw=2,
                label=f'μ_max={fi["mu_max"]:.2f}, β={fi["beta"]:.4f}')
        for xi,yi in zip(fi["x"][si],fi["y"][si]):
            ax.annotate(f"{yi:.1f}",(xi,yi),textcoords="offset points",xytext=(0,12),fontsize=8,ha="center")
        ax.set_title(fi["label"].replace("_"," ").title())
        ax.set_xlabel("Concurrency" if "concurrency" in fi["label"] else "Memory Pressure x")
        ax.set_ylabel("Decode Rate (tok/s)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.text(0.97,0.97,f'R²={fi["r2"]:.4f}\nRMSE={fi["rmse_val"]:.2f}',
                transform=ax.transAxes, fontsize=10, va="top", ha="right",
                bbox=dict(boxstyle="round",fc="wheat",alpha=0.8))
    plt.tight_layout()
    plt.savefig(f"{a.out_prefix}_plot.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {a.out_prefix}_plot.png\nDone!")

if __name__ == "__main__":
    main()
