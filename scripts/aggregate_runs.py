import argparse, json, os, re, sys
import numpy as np, pandas as pd
from aggregate_results import parse_vllm_result, parse_gpu_log

def aggregate_all_runs(run_dirs, out_summary_runs, out_summary_means):
    concurrency_levels = [1, 2, 4, 8, 16, 32]
    all_runs_data = {r: {} for r in run_dirs}
    for run_dir in run_dirs:
        results_dir = os.path.join(run_dir, 'results')
        gpu_log_dir = os.path.join(run_dir, 'gpu_logs')
        if not os.path.exists(results_dir):
            print(f"Warning: {results_dir} does not exist, skipping.")
            continue
        rfs = sorted([f for f in os.listdir(results_dir) if f.startswith('result_concurrency_') and f.endswith('.json')])
        for rf in rfs:
            row = parse_vllm_result(os.path.join(results_dir, rf))
            gl = os.path.join(gpu_log_dir, f"gpu_concurrency_{row['concurrency']}.csv")
            if os.path.exists(gl):
                row.update(parse_gpu_log(gl))
            if 'tpot_ms' in row and row['tpot_ms'] > 0:
                row['decode_rate_tok_s'] = round(1000.0 / row['tpot_ms'], 4)
            all_runs_data[run_dir][row['concurrency']] = row
    aggregated_rows = []
    means_rows = []
    numeric_cols = [
        'tpot_ms', 'decode_rate_tok_s', 'ttft_ms', 'throughput_tok_s', 
        'e2e_latency_ms', 'gpu_mem_used_mib_mean', 'gpu_mem_used_mib_max',
        'gpu_mem_total_mib', 'mem_pressure_x_mean', 'mem_pressure_x_max',
        'gpu_util_pct_mean', 'power_w_mean', 'temp_c_mean'
    ]
    for C in concurrency_levels:
        run_values = {col: [] for col in numeric_cols}
        for run_dir in run_dirs:
            if C in all_runs_data[run_dir]:
                row = all_runs_data[run_dir][C]
                for col in numeric_cols:
                    if col in row and row[col] is not None:
                        run_values[col].append(row[col])
        agg_row = {'concurrency': C}
        mean_row = {'concurrency': C}
        for col in numeric_cols:
            vals = run_values[col]
            if len(vals) > 0:
                mean_val = np.mean(vals)
                std_val = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                agg_row[f'{col}_mean'] = round(mean_val, 4)
                agg_row[f'{col}_std'] = round(std_val, 4)
                mean_row[col] = round(mean_val, 4)
            else:
                agg_row[f'{col}_mean'] = None
                agg_row[f'{col}_std'] = None
                mean_row[col] = None
        aggregated_rows.append(agg_row)
        means_rows.append(mean_row)
    df_agg = pd.DataFrame(aggregated_rows)
    df_means = pd.DataFrame(means_rows)
    df_agg.to_csv(out_summary_runs, index=False)
    df_means.to_csv(out_summary_means, index=False)
    print(f"Saved aggregated runs to: {out_summary_runs}")
    print(f"Saved compatible means to: {out_summary_means}")
    print("\n--- Aggregated Concurrency Table (Mean +/- Std) ---")
    print(f"{'C':<4} | {'TPOT (ms)':<18} | {'Decode Rate (tok/s)':<22} | {'Throughput (tok/s)':<22}")
    print("-" * 75)
    for row in aggregated_rows:
        tpot_str = f"{row['tpot_ms_mean']:.2f} +/- {row['tpot_ms_std']:.2f}"
        rate_str = f"{row['decode_rate_tok_s_mean']:.2f} +/- {row['decode_rate_tok_s_std']:.2f}"
        thru_str = f"{row['throughput_tok_s_mean']:.2f} +/- {row['throughput_tok_s_std']:.2f}"
        print(f"{row['concurrency']:<4} | {tpot_str:<18} | {rate_str:<22} | {thru_str:<22}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True, help='Run directories to aggregate')
    parser.add_argument('--out-summary-runs', default='calibration_summary_runs.csv')
    parser.add_argument('--out-summary-means', default='calibration_summary.csv')
    args = parser.parse_args()
    aggregate_all_runs(args.runs, args.out_summary_runs, args.out_summary_means)
