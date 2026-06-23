# ReaLM-Retrieve

### When to Retrieve **During** Reasoning — Adaptive RAG for Large Reasoning Models

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg?style=flat-square)](https://github.com/bettyguo/realm-retrieve/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)

---

## How it works

```
                ┌─────────────────────────────────────────┐
                │             user question               │
                └──────────────────┬──────────────────────┘
                                   v
              ┌────────────────────────────────────────────┐
              │   Large Reasoning Model (DeepSeek-R1 ...)  │
              │       generates extended chain-of-thought  │
              └──────────────────┬─────────────────────────┘
                                 v
            1. StepBoundaryClassifier   (boundary detection)
                                 v
            2. RSUS  =  a*U_verb + b*U_ent + g*U_cons     <-- per step
                                 v
            3. Policy(retrieve | state)  (REINFORCE, lambda-curriculum)
                    |                          |
                   skip                     retrieve
                    |                          v
                    |              QueryGen -> ColBERTv2/PLAID -> evidence
                    |                          v
                    |              ImplicitCompression
                    |                          v
                    └────────> RetrievalInjector (KV-cache-aware) ────>
                                               v
                                        final answer
```

---

## Architecture

| Component | File | Description |
|---|---|---|
| **Segmenter** | [segmentation.py](src/realm_retrieve/models/segmentation.py) | Transformer splitting reasoning chains into logical steps |
| **RSUS calculator** | [rsus.py](src/realm_retrieve/models/rsus.py) | Composite uncertainty: U_verb + U_ent + U_cons with learned weights |
| **Policy network** | [policy.py](src/realm_retrieve/models/policy.py) | 4-layer transformer encoder; binary retrieve/continue decision |
| **QueryGen** | [query_gen.py](src/realm_retrieve/models/query_gen.py) | Transformer decoder; formulates dense queries from reasoning context |
| **Proxy MLP** | [proxy_mlp.py](src/realm_retrieve/models/proxy_mlp.py) | MLP for uncertainty estimation on completion-only models |
| **Retriever** | [retriever.py](src/realm_retrieve/models/retriever.py) | ColBERTv2 + PLAID late-interaction search |
| **Compression** | [compression.py](src/realm_retrieve/models/compression.py) | Attention-weighted sentence filtering |
| **Speculative cache** | [speculative_cache.py](src/realm_retrieve/models/speculative_cache.py) | Entity-based parallel pre-fetching |
| **KV-cache manager** | [kv_cache.py](src/realm_retrieve/models/kv_cache.py) | Prefix-aware evidence injection preserving vLLM KV states |
| **Injection protocol** | [injection.py](src/realm_retrieve/models/injection.py) | Full injection pipeline |
| **LRM adapter** | [reasoning_model.py](src/realm_retrieve/models/reasoning_model.py) | Unified API for DeepSeek-R1, QwQ, o1 |
| **Metrics** | [metrics.py](src/realm_retrieve/evaluation/metrics.py) | EM, F1, Sup-F1, Evi-F1, retrieval quality, efficiency, paired bootstrap |

---

## Project layout

```
realm-retrieve/
├── src/realm_retrieve/           # installable package
│   ├── models/                   #   all model components (see table above)
│   └── evaluation/               #   QA + retrieval + efficiency + bootstrap metrics
├── configs/
│   └── experiments/              #   Hydra configs for training
├── scripts/
│   ├── training/                 #   full training scripts (segmenter, policy, proxy MLP, RSUS calibration)
│   └── evaluation/               #   evaluation pipeline
├── evaluate.py                   #   quick evaluation entry point
├── train_segmentation.py         #   segmenter training entry point
├── train_policy.py               #   policy training entry point
└── REPRODUCE.md                  #   step-by-step reproduction guide
```

---

## Quick start

```bash
# Install
git clone https://github.com/bettyguo/realm-retrieve.git
cd realm-retrieve
pip install -e ".[dev]"
```

## Training

```bash
# Step 1: Train step-boundary classifier
python scripts/training/train_segmenter_full.py \
    --data data/annotations/nq_train_traces.jsonl \
    --test_data data/annotations/nq_test_traces.jsonl \
    --output checkpoints/segmentation/ --seed 42

# Step 2: Train proxy MLP for completion-only models
python scripts/training/train_proxy_mlp.py \
    --data data/annotations/nq_train_traces.jsonl \
    --output checkpoints/proxy_mlp/ --seed 42

# Step 3: Calibrate RSUS weights and thresholds
python scripts/training/calibrate_rsus.py \
    --data data/processed/musique/dev.jsonl \
    --segmenter_checkpoint checkpoints/segmentation/best_model/ \
    --output rsus_calibration/ --mode all

# Step 4: Train intervention policy + QueryGen via REINFORCE
python scripts/training/train_policy_full.py \
    --train_data data/processed/musique/train.jsonl \
    --dev_data data/processed/musique/dev.jsonl \
    --segmenter_checkpoint checkpoints/segmentation/best_model/ \
    --output checkpoints/policy/ --seed 42

# Multi-seed runs (seeds 42, 123, 456)
python scripts/training/multi_seed_runner.py --component segmenter --seeds 42,123,456
python scripts/training/multi_seed_runner.py --component policy --seeds 42,123,456
```

## Evaluation

```bash
# Quick evaluation via Hydra config (paths configured in configs/experiments/evaluate.yaml)
python evaluate.py dataset=musique

# Full live GPU evaluation (requires trained checkpoints + index)
python scripts/evaluation/evaluate_full.py \
    --method realm_retrieve --dataset musique \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --seed 42 \
    --data data/processed/musique/dev.jsonl \
    --segmenter_checkpoint checkpoints/segmentation/best_model/ \
    --policy_checkpoint checkpoints/policy/best_model.pt \
    --index_path data/indices/colbert/musique.plaid \
    --output outputs/results/

# Recompute metrics from existing prediction files (no GPU, no inference)
python scripts/evaluation/evaluate_full.py \
    --method realm_retrieve --dataset musique \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --seed 42 \
    --recompute-metrics predictions/main_results/ \
    --output outputs/recomputed/
```

---

## Reproduction

See [REPRODUCE.md](REPRODUCE.md) for the full step-by-step guide covering data download, index construction, training, and evaluation.

---

Released under the [Apache 2.0 License](LICENSE).
