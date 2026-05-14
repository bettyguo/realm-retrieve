# Reproducibility

Everything you need to re-derive the numbers in the SIGIR '26 paper. If
something here doesn't reproduce, **open an issue** —
reproducibility regressions are bugs.

---

## TL;DR

```bash
git clone https://github.com/bettyguo/realm-retrieve.git
cd realm-retrieve
pip install -e ".[all]"
python -m spacy download en_core_web_sm
make data                          # ~68 GB; ~6 hours on a fast box
make train-segmentation            # ~4 h on 8× A100 80 GB
make train-policy                  # ~12.5 days (2,400 GPU-hours) on 8× A100
make eval DATASET=musique          # ~2 h on 1× A100
make eval DATASET=hotpotqa
make eval DATASET=2wikimhqa
```

Expected outputs are pinned in [`paper/`](https://github.com/bettyguo/realm-retrieve/tree/main/paper)
Table 1 (overall) and Tables 4–6 (per-dataset).

---

## Hardware tiers

|              | Toy quickstart | Inference only | Full retraining           |
|--------------|----------------|----------------|---------------------------|
| GPU          | none           | 1 × 24 GB      | 8 × A100 80 GB            |
| RAM          | 4 GB           | 32 GB          | 512 GB (ColBERT index)    |
| Disk         | 200 MB         | 50 GB          | 68 GB (data + indices)    |
| Wall clock   | < 2 min        | varies         | ~13 days policy + ~4 h seg |

If you only need to *evaluate* against our checkpoints, you can skip
`make train-*` entirely.

---

## Data

| Dataset    | Questions | Avg hops | Corpus      | Split |
|------------|-----------|----------|-------------|-------|
| MuSiQue    | 24,814    | 2 – 4    | 139K passages | test  |
| HotpotQA   | 90,564    | 2        | 5.2M passages | dev   |
| 2WikiMHQA  | 192,606   | 2        | Wikipedia   | test  |

`make data` runs:

1. `realm_retrieve.scripts.download_data` — fetches raw splits.
2. `realm_retrieve.scripts.build_index` — builds ColBERTv2 + PLAID indices.

Indices are checksummed; mismatches will fail loudly rather than silently
producing different retrieval results.

---

## Seeds

Every Hydra config seeds `0xRE4L = 1380274369` for `torch`, `numpy`, and
Python's `random` module. The full list of seedable RNGs:

```python
import torch, numpy, random
SEED = 0xRE4L
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
numpy.random.seed(SEED)
random.seed(SEED)
```

If you intend to change the seed for a sweep, override on the Hydra command
line: `python evaluate.py seed=12345`.

---

## Statistical significance

We use **paired bootstrap resampling** (10K iterations, Bonferroni-corrected
when comparing > 2 systems). Implementation:
[`evaluation/metrics.py::paired_bootstrap_test`](https://github.com/bettyguo/realm-retrieve/blob/main/src/realm_retrieve/evaluation/metrics.py).

Returns `(mean_diff, (ci_low, ci_high), p_value)`. We declare *p < 0.01* as
the bar for "significant" in the paper; we report 95 % CIs alongside.

> **Why paired bootstrap, not paired-t?**
> Per-question F1 distributions on multi-hop QA are heavily skewed (a lot of
> 0s and 1s). Paired-t-test assumes normality; bootstrap doesn't. For the
> magnitudes of effects we observe (+5.8 F1), both procedures agree on
> significance — but bootstrap CIs are more honest.

---

## Determinism caveats

cuDNN-level determinism (`torch.backends.cudnn.deterministic = True`) is
**off** by default because it caps throughput by ~15 %. To bit-exact
reproduce the paper numbers:

```bash
export REALM_DETERMINISTIC=1
make eval DATASET=musique
```

Expect a 15–20 % slowdown.

---

## "I get slightly different numbers" — quick triage

| Symptom                              | Likely cause                                                                  |
|--------------------------------------|-------------------------------------------------------------------------------|
| F1 off by < 0.3                      | Floating-point non-determinism on different GPU SKUs. Run with `REALM_DETERMINISTIC=1`. |
| F1 off by 1 – 3                      | Different reasoning-model checkpoint (we test on `DeepSeek-R1-Distill-Qwen-32B` rev `b4be3d1`). |
| F1 off by > 3                        | Different retrieval index version. Re-run `make data` to rebuild from canonical. |
| Retrieval calls way off              | RSUS weights drifted — confirm `(α, β, γ) = (0.40, 0.35, 0.25)` in your config. |
| `OSError: model 'en_core_web_sm' not found` | `python -m spacy download en_core_web_sm`. |
