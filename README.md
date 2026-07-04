# vLLM Service Calibration Sweeps: Cross-Hardware Validation

An empirical cross-hardware benchmarking and calibration repository investigating LLM serving performance under varying concurrency loads. This project contains both the orchestration codebase and the validation dataset collected on **NVIDIA RTX 3090** (Ampere) and **NVIDIA RTX 4090** (Ada Lovelace) GPUs.

The goal is to evaluate the consistency of the exponential service-degradation model under load across different models and hardware setups.

---

## 📊 Overview & Core Models

This repository supports two distinct calibration formulations to model performance degradation:

### 1. Primary Calibration Fit

Models the service rate as a function of the internal vLLM GPU KV-cache usage percentage ($x$):

$$\mu_{\text{emp}}(x) = 53.61 \cdot e^{-0.0344 x}$$

where:
* **$\mu_{\text{emp}}(x)$** is the decode rate (tokens/s), defined as $\frac{1000}{\text{TPOT}}$ (Time Per Output Token in ms).
* **$x$** is the GPU KV-cache usage in percent.

This formulation represents the primary calibration of the manuscript, yielding $R^2 = 0.996$ and $\text{RMSE} = 0.79 \text{ tok/s}$.

### 2. Secondary Robustness Proxies (Cross-Model & Cross-Hardware)

Model the service rate as a function of request concurrency ($C$):

$$\mu(C) = \mu_{\max} \cdot e^{-\beta C}$$

where:
* **$C$** is the request concurrency.
* **$\mu_{\max}$** is the maximum decode rate (tokens/s).
* **$\beta$** is the degradation parameter.

> [!NOTE]
> The GPU memory logs for the other models and hardware are outside the scope of this repository. The robustness fits use load proxies because the corresponding runs did not retain the same internal KV-cache telemetry.

---

## 📋 Relationship to the Manuscript

The following table maps the manuscript's calibration and robustness goals to the repository's directories and scripts:

| Manuscript Goal | Calibration Type | Repository Folder | Plot/Output File | Execution Script |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Calibration** | Qwen2.5-7B / RTX 3090 using vLLM internal GPU KV-cache usage (%) | `data/rtx_3090/qwen2.5_7b_instruct/` | `outputs/qwen_3090_kv_cache_fit.png` | `scripts/fit_kv_cache.py` |
| **Cross-model robustness** | Mistral-7B / RTX 3090 using estimated resident-token pressure proxy or concurrency proxy | `data/rtx_3090/mistral_7b_instruct_v0.3/` | `outputs/mistral_3090_fit.png` | `scripts/fit_concurrency.py` |
| **Cross-hardware robustness** | Qwen2.5-7B / RTX 4090 using concurrency pressure proxy | `data/rtx_4090/qwen2.5_7b_instruct/` | `outputs/qwen_4090_concurrency_fit.png` | `scripts/fit_concurrency.py` |

---

## 📈 Cross-Hardware Performance Summary

The experiments were run with **Qwen2.5-7B-Instruct** (precision `float16`, max model length `4096`, memory utilization `0.85`, random prompts with 1024 input/128 output tokens) and **Mistral-7B-Instruct-v0.3** across concurrency levels $C \in \{1, 2, 4, 8, 16, 32\}$ (averaged over 3 sweeps of 256 prompts each per concurrency level, for a total of 4,608 requests per sweep campaign).

### Exponential Model Fits

| Hardware | Model | Load Variable | $\mu_{\max}$ (tok/s) | $\beta$ (degradation coeff) | $R^2$ | RMSE (tok/s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **RTX 3090 (Ampere)** | Qwen2.5-7B-Instruct | GPU KV-cache usage (%) | $53.61$ | $0.0344$ | $0.9960$ | $0.79$ |
| **RTX 3090 (Ampere)** | Mistral-7B-Instruct-v0.3 | Concurrency | $54.33$ | $0.0514$ | $0.9772$ | $2.14$ |
| **RTX 4090 (Ada Lovelace)** | Qwen2.5-7B-Instruct | Concurrency | $64.00$ | $0.0280$ | $0.9934$ | $1.03$ |

### Key Hardware Insights

1. **Single-User Throughput Gain ($\mu_{\max}$)**:
   * Upgrading from RTX 3090 to RTX 4090 yields a speedup in raw maximum decode throughput (from $53.04 \text{ tok/s}$ to $64.00 \text{ tok/s}$ concurrency-based, and $53.61 \text{ tok/s}$ cache-based).

2. **Concurrency Resilience ($\beta$)**:
   * The degradation parameter $\beta$ is lower on the RTX 4090 compared to the RTX 3090. This suggests that the RTX 4090 configuration exhibits a flatter response curve under the tested workload.

---

## 📂 Repository Structure

```
vllm-calibration-sweeps/
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── rtx_3090/
│   │   ├── qwen2.5_7b_instruct/
│   │   │   ├── raw/                  # Raw benchmark json files and server logs
│   │   │   ├── processed/            # Aggregated performance data CSVs
│   │   │   └── fit/                  # Generated fit parameters and plots
│   │   └── mistral_7b_instruct_v0.3/
│   │       ├── raw/
│   │       ├── processed/
│   │       └── fit/
│   └── rtx_4090/
│       └── qwen2.5_7b_instruct/
│           ├── raw/
│           ├── processed/
│           └── fit/
├── scripts/
│   ├── setup.sh                      # Environment configuration & library installs
│   ├── start_server.sh               # Orchestrates local vLLM server launch
│   ├── run_sweep.sh                  # Executes the concurrency benchmark sweeps
│   ├── parse_qwen_3090.py            # Extracts Qwen 3090 KV cache usage
│   ├── parse_mistral_3090.py         # Extracts Mistral 3090 decode rates
│   ├── parse_qwen_4090.py            # Extracts Qwen 4090 decode rates
│   ├── fit_kv_cache.py               # Fits the primary KV-cache model
│   ├── fit_concurrency.py            # Fits concurrency-proxy models
│   └── plot_all_fits.py              # Generates publication plots and consolidated summary
├── solver/
│   └── model_verification.py         # Checks model outputs against manuscript specs
└── outputs/                          # Consolidated publication outputs and plots
```

---

## ⚙️ How to Reproduce & Run Sweeps

All scripts are located in the `scripts/` and `solver/` folders.

### 1. Environment Setup

Run the setup script to install necessary Python and system dependencies:

```bash
bash scripts/setup.sh
```

### 2. Parse Raw Data and Generate Fits

Run the following scripts to parse the raw data, execute the mathematical fits, output plots, and verify correctness against the manuscript:

```bash
# 1. Parse raw data into processed CSV tables
python3 scripts/parse_qwen_3090.py
python3 scripts/parse_mistral_3090.py
python3 scripts/parse_qwen_4090.py

# 2. Fit primary KV-cache model
python3 scripts/fit_kv_cache.py

# 3. Fit secondary robustness models
python3 scripts/fit_concurrency.py \
    --input data/rtx_3090/mistral_7b_instruct_v0.3/processed/mistral_3run_summary.csv \
    --out data/rtx_3090/mistral_7b_instruct_v0.3/fit/mistral_concurrency_fit_summary.txt \
    --label "Mistral-7B / RTX 3090 concurrency robustness fit"

python3 scripts/fit_concurrency.py \
    --input data/rtx_4090/qwen2.5_7b_instruct/processed/qwen_4090_summary.csv \
    --out data/rtx_4090/qwen2.5_7b_instruct/fit/qwen_4090_concurrency_fit_summary.txt \
    --label "Qwen2.5-7B / RTX 4090 concurrency robustness fit"

# 4. Generate consolidated publication plots and summary table
python3 scripts/plot_all_fits.py

# 5. Run the verification script to check constraints
python3 solver/model_verification.py
```

---

## 📝 Empirical Verification Output

Running the mathematical fit and verification suite locally produces the following validated regression output:

```
mu_max=53.6163 tok/s
beta=0.034410 per KV-cache percentage point
R2=0.996001
RMSE=0.790135 tok/s
Wrote data/rtx_3090/qwen2.5_7b_instruct/fit/qwen_3090_kv_cache_fit_summary.txt
Generated qwen_3090_kv_cache_fit.png
Generated mistral_3090_fit.png
Generated qwen_4090_concurrency_fit.png
Wrote performance summary to outputs/model_hardware_robustness_summary.csv
==================================================
Qwen2.5-7B / RTX 3090 (KV-cache usage % fit):
  mu_max = 53.6163 (Expected ≈ 53.61)
  beta   = 0.0344 (Expected ≈ 0.0344)
  R2     = 0.9960 (Expected >= 0.99)
  RMSE   = 0.7901 (Expected ≈ 0.79)
  [PASS] All parameters match manuscript specifications.
--------------------------------------------------
Mistral-7B / RTX 3090 (concurrency fit):
  R2 = 0.9772 (Expected >= 0.97)
  [PASS] R2 matches manuscript specifications.
--------------------------------------------------
Qwen2.5-7B / RTX 4090 (concurrency fit):
  R2 = 0.9934 (Expected >= 0.99)
  [PASS] R2 matches manuscript specifications.
==================================================
Overall verification result: SUCCESS
```
