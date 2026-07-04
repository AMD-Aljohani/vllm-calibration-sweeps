# Validation Output

These commands were run before creating the Zenodo release for the MDPI Systems manuscript submission.

## Execution Command:
```bash
python3 scripts/fit_kv_cache.py

python3 scripts/fit_concurrency.py \
  --input data/rtx_4090/qwen2.5_7b_instruct/processed/qwen_4090_summary.csv \
  --out data/rtx_4090/qwen2.5_7b_instruct/fit/qwen_4090_concurrency_fit_summary.txt \
  --label "Qwen2.5-7B / RTX 4090 concurrency-proxy robustness fit"

python3 solver/model_verification.py validate
python3 solver/model_verification.py optimize
```

## Terminal Output:
```
mu_max=53.6163 tok/s
beta=0.034410 per KV-cache percentage point
R2=0.996001
RMSE=0.790135 tok/s
Wrote data/rtx_3090/qwen2.5_7b_instruct/fit/qwen_3090_kv_cache_fit_summary.txt
[Qwen2.5-7B / RTX 4090 concurrency-proxy robustness fit]
mu_max=63.9993 tok/s
beta=0.028015 per concurrency point
R2=0.993391
RMSE=1.024056 tok/s
Wrote summary to data/rtx_4090/qwen2.5_7b_instruct/fit/qwen_4090_concurrency_fit_summary.txt
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
