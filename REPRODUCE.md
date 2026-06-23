# ReaLM-Retrieve: Full Reproduction Guide

> Reproducing results from **"When to Retrieve During Reasoning"** (ReaLM-Retrieve)

---

## 1. Overview

| Path | Hardware | Wall-clock | What it validates |
|------|----------|------------|-------------------|
| **Full reproduction** | 8x A100 80GB | ~2 weeks | End-to-end training + evaluation from scratch |

**Time breakdown (full reproduction):**
- Data download & preprocessing: ~3 hours
- Index construction: ~3 hours
- Training (all components, 3 seeds): ~2,400 GPU-hours
- Evaluation (all methods, 3 seeds): ~200 GPU-hours

---

## 2. Prerequisites

### Hardware

| Tier | GPUs | Use case |
|------|------|----------|
| **Full** | 8x NVIDIA A100 80GB | Training + evaluation |
| **Inference-only** | 1x A100 80GB | Run evaluations with existing checkpoints |
| **R1-671B** | DeepSeek API or 8x H100 | Full-size model evaluation |

### Software

- Python 3.10+ (tested on 3.10, 3.11)
- CUDA 12.1+ (for GPU paths)
- PyTorch 2.2-2.4
- vLLM 0.3-0.5 (for model serving)
- ~200 GB disk (datasets ~50 GB, indices ~80 GB, checkpoints ~30 GB, outputs ~40 GB)
- 256 GB+ system RAM recommended for ColBERT indexing

### API Keys (optional)

- `DEEPSEEK_API_KEY` — required only for R1-671B API evaluation
- `OPENAI_API_KEY` — required only for o1 evaluation
- `HF_TOKEN` — for gated HuggingFace models
- `WANDB_API_KEY` — optional experiment tracking

---

## 3. Environment Setup

```bash
# 1. Clone and install
git clone https://github.com/bettyguo/realm-retrieve.git
cd realm-retrieve
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

pip install -e ".[dev]"

# 2. Install retrieval and training dependencies
pip install -e ".[serve,train]"
pip install colbert-ai sentence-transformers spacy
python -m spacy download en_core_web_sm
```

---

## 4. Data Download and Preprocessing (~2 hours)

Download MuSiQue, HotpotQA, and 2WikiMultiHopQA, then preprocess into JSONL format. Build ColBERT/PLAID indices for retrieval (~1 hour per dataset on 1 GPU).

---

## 5. Training

### 5a. Segmenter (~2 hours on 1 GPU)

```bash
python scripts/training/train_segmenter_full.py \
    --data data/annotations/nq_train_traces.jsonl \
    --test_data data/annotations/nq_test_traces.jsonl \
    --output checkpoints/segmentation/ \
    --seed 42
```

### 5b. Policy + QueryGen

```bash
python scripts/training/train_policy_full.py \
    --train_data data/processed/musique/train.jsonl \
    --dev_data data/processed/musique/dev.jsonl \
    --segmenter_checkpoint checkpoints/segmentation/best_model/ \
    --output checkpoints/policy/ \
    --seed 42
```

For full LLM-based reward with live RSUS computation, add `--use_online_f1`:

```bash
python scripts/training/train_policy_full.py \
    --train_data data/processed/musique/train.jsonl \
    --dev_data data/processed/musique/dev.jsonl \
    --segmenter_checkpoint checkpoints/segmentation/best_model/ \
    --output checkpoints/policy/ \
    --seed 42 \
    --use_online_f1 \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --index_path data/indices/colbert/musique.plaid
```

### 5c. Proxy MLP (~30 minutes on 1 GPU)

```bash
python scripts/training/train_proxy_mlp.py \
    --data data/annotations/nq_train_traces.jsonl \
    --test_data data/annotations/nq_test_traces.jsonl \
    --output checkpoints/proxy_mlp/ \
    --seed 42
```

### 5d. RSUS Calibration (~10 hours on 1 GPU)

```bash
python scripts/training/calibrate_rsus.py \
    --data data/processed/musique/dev.jsonl \
    --segmenter_checkpoint checkpoints/segmentation/best_model/ \
    --output rsus_calibration/ \
    --mode all
```

### 5e. Multi-seed Runs

```bash
python scripts/training/multi_seed_runner.py --component segmenter --seeds 42,123,456
python scripts/training/multi_seed_runner.py --component policy --seeds 42,123,456
```

---

## 6. Evaluation

### 6a. Live GPU Evaluation

```bash
for dataset in musique hotpotqa 2wikimhqa; do
    for seed in 42 123 456; do
        python scripts/evaluation/evaluate_full.py \
            --method realm_retrieve \
            --dataset ${dataset} \
            --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
            --seed ${seed} \
            --data data/processed/${dataset}/dev.jsonl \
            --segmenter_checkpoint checkpoints/segmentation/best_model/ \
            --policy_checkpoint checkpoints/policy/best_model.pt \
            --index_path data/indices/colbert/${dataset}.plaid \
            --output outputs/results/
    done
done
```

### 6b. Recompute Metrics (No GPU)

To recompute EM/F1/RSUS from existing prediction files without re-running
inference (e.g. after a metric-formula update):

```bash
for dataset in musique hotpotqa 2wikimhqa; do
    for seed in 42 123 456; do
        python scripts/evaluation/evaluate_full.py \
            --method realm_retrieve \
            --dataset ${dataset} \
            --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
            --seed ${seed} \
            --recompute-metrics predictions/main_results/ \
            --output outputs/recomputed/
    done
done
```

This mode does **not** load any model checkpoint or run any inference.
It reads the `predicted_answer` and `gold_answer` fields from existing JSONL
records and recomputes scores.  Use Section 6a for full model evaluation.

---

## 7. Embedding Dimensions and Model Consistency

All components that produce or consume embeddings for the retrieval policy use
`sentence-transformers/all-mpnet-base-v2` (768-dimensional):

| Component | Encoder | Dimension |
|-----------|---------|-----------|
| Training (`train_policy_full.py`) | `all-mpnet-base-v2` | 768 |
| Evaluation (`evaluate.py`, `evaluate_full.py`) | `all-mpnet-base-v2` (via `reasoning_model.compute_embedding`) | 768 |
| Policy model default | `embedding_dim=768` | 768 |
| Configs (`train_policy.yaml`, `evaluate.yaml`) | `embedding_dim: 768` | 768 |
| QueryGen output | `output_dim=768` (ColBERTv2 query space) | 768 |

The checkpoint loader validates embedding dimensions at load time and raises a
clear error if the checkpoint was trained with a different dimension than the
model being loaded into.

**Note:** Other components (segmenter, RSUS calculator, implicit compression)
use `all-MiniLM-L6-v2` (384-d) independently for their own internal
computations. These do not interact with the policy's embedding space.

---

## 8. Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM during vLLM serving | Reduce `max_model_len` or `gpu_memory_utilization` in vLLM config |
| ColBERT indexing fails | Ensure corpus TSV format: `pid\ttext\ttitle` (tab-separated) |
| Policy training diverges | Reduce learning rate from 1e-4 to 5e-5, increase warmup steps |
| spaCy model missing | Run `python -m spacy download en_core_web_sm` |
| DeepSeek API rate limit | Add `--rate_limit 5` flag or reduce batch size |
| Large memory during indexing | ColBERT indexing needs ~256 GB RAM for HotpotQA (5.2M passages) |
