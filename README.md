# vLLM Service Calibration Sweeps: Cross-Hardware Validation

An empirical cross-hardware benchmarking and calibration repository investigating LLM serving performance under varying concurrency loads. This project contains both the orchestration codebase and the validation dataset collected on **NVIDIA RTX 3090** (Ampere) and **NVIDIA RTX 4090** (Ada Lovelace) GPUs.

---

## 📊 Overview & Core Model

The core objective is to validate the universality of the exponential service-degradation model under load. 

The decode service rate $\mu(C)$ (tokens/sec) degrades exponentially as a function of the concurrency level $C$:

$$\mu(C) = \mu_{\max} \cdot e^{-\beta C}$$

where:
* **$\mu(C)$** is the decode service rate (tokens/sec), defined as $\frac{1000}{\text{TPOT}}$ (Time Per Output Token in ms).
* **$\mu_{\max}$** is the theoretical maximum decode speed (single-user service rate).
* **$\beta$** is the service degradation parameter under concurrency load.
* **$C$** is the client concurrency level.

---

## 📈 Cross-Hardware Performance Summary

The experiments were run with **Qwen2.5-7B-Instruct** (precision `float16`, max model length `4096`, memory utilization `0.85`, random prompts with 1024 input/128 output tokens) and **Mistral-7B-Instruct-v0.3** across concurrency levels $C \in \{1, 2, 4, 8, 16, 32\}$ (averaged over 3 sweeps of 256 prompts each per concurrency level, for a total of 4,608 requests per sweep campaign).

### Exponential Model Fits:

| Hardware | Model | $\mu_{\max}$ (tok/s) | $\beta$ (degradation coeff) | $R^2$ | RMSE (tok/s) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **RTX 3090 (Ampere)** | Qwen2.5-7B-Instruct | $53.04$ | $0.0403$ | $0.9911$ | $1.20$ |
| **RTX 3090 (Ampere)** | Mistral-7B-Instruct | $54.33$ | $0.0514$ | $0.9772$ | $2.14$ |
| **RTX 4090 (Ada Lovelace)** | Qwen2.5-7B-Instruct | $64.00$ | $0.0280$ | $0.9934$ | $1.03$ |

### Key Hardware Insights:

1. **Single-User Throughput Gain ($\mu_{\max}$)**:
   * Upgrading from RTX 3090 to RTX 4090 yields a **$20.7\%$ speedup** in raw maximum decode throughput (from $53.04 \text{ tok/s}$ to $64.00 \text{ tok/s}$).

2. **Concurrency Resilience ($\beta$)**:
   * The degradation parameter $\beta$ is **$30.5\%$ lower** on the RTX 4090 compared to the RTX 3090 (0.0280 vs 0.0403). This confirms that the newer Ada Lovelace architecture (specifically memory bandwidth and cache behavior under vLLM's PagedAttention scheduling) maintains a much flatter response curve under high request rates.

---

## 📂 Repository Structure

```tree
vllm-calibration-sweeps/
├── LICENSE
├── README.md
├── requirements.txt
├── scripts/
│   ├── setup.sh                     # Environment configuration & library installs
│   ├── start_server.sh              # Orchestrates local vLLM server launch
│   ├── run_sweep.sh                 # Executes the concurrency benchmark sweeps
│   ├── aggregate_results.py         # Parses single-run JSONs and CSV telemetry
│   ├── aggregate_runs.py            # Combines and averages multiple runs
│   └── fit_degradation_model.py     # Fits exponential model and generates plots
└── data/
    ├── rtx_3090/
    │   ├── qwen2.5_7b_instruct/     # Raw runs, aggregate CSVs, fits, & plots for Qwen
    │   └── mistral_7b_instruct/     # Raw runs, aggregate CSVs, fits, & plots for Mistral
    └── rtx_4090/
        └── qwen2.5_7b_instruct/     # Raw runs, aggregate CSVs, fits, & plots for Qwen
```

---

## ⚙️ How to Reproduce & Run Sweeps

All scripts are located in the `scripts/` folder. The environment should be configured on a CUDA-enabled instance (e.g., RunPod, Lambdalabs, local station).

### 1. Environment Setup
Run the setup script to install necessary Python and system dependencies:
```bash
bash scripts/setup.sh
```

### 2. Start the vLLM Server
Launch the model server in a detached screen or terminal tab:
```bash
# Starts Qwen2.5-7B-Instruct on port 8000
bash scripts/start_server.sh Qwen/Qwen2.5-7B-Instruct 4096
```

### 3. Execute Calibration sweeps
In another terminal, run the concurrency sweep:
```bash
# Sweeps through concurrency levels [1, 2, 4, 8, 16, 32]
# Saves output results and nvidia-smi GPU logs locally
bash scripts/run_sweep.sh qwen7b
```

### 4. Post-processing and Fitting
Aggregate single runs, run statistics across repeated campaigns, and fit the exponential decay degradation model:

```bash
# 1. Aggregate results from multiple runs (e.g. run_1, run_2, run_3)
python3 scripts/aggregate_runs.py \
    --runs run_1 run_2 run_3 \
    --out-summary-runs calibration_summary_runs.csv \
    --out-summary-means calibration_summary.csv

# 2. Fit the exponential decay model and plot results
python3 scripts/fit_degradation_model.py \
    --csv calibration_summary.csv \
    --out-prefix fit_output
```

---

## 📈 Parameter & Plot Outputs

Inside the respective `data/` directories, you will find:
* `metadata.txt`: Full hardware & engine configurations.
* `calibration_summary.csv`: Averaged measurements per concurrency level.
* `fit_output_plot.png`: Fitted exponential curve overlaying measured data.
* `fit_output_parameters.txt`: Calculated regression coefficients ($\mu_{\max}, \beta$) and confidence intervals.
