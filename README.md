# ReaLM-Retrieve: When to Retrieve During Reasoning

<div align="center">

**Adaptive Retrieval for Large Reasoning Models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## Overview

Large reasoning models such as DeepSeek-R1 and OpenAI o1 generate extended chains of thought spanning thousands of tokens, yet their integration with retrieval-augmented generation (RAG) remains fundamentally misaligned. **ReaLM-Retrieve** addresses this mismatch through three key innovations:

1. **Step-Level Uncertainty Detection (RSUS)**: Identifies knowledge gaps at reasoning-step granularity rather than token or sentence level
2. **Learned Retrieval Intervention Policy**: REINFORCE-trained policy that learns when external evidence maximally benefits ongoing reasoning
3. **Efficiency-Optimized Integration**: Reduces per-retrieval overhead by 3.2× through implicit decompression and speculative caching

### Key Results

On **MuSiQue** multi-hop QA benchmark:
- **71.2% F1** with only **1.8 retrieval calls** per question
- **+5.8% F1** absolute improvement over IRCoT baseline
- **47% fewer retrieval calls** than fixed-interval approaches
- **2.1× better accuracy-efficiency ratio**

All improvements statistically significant at p < 0.01 (paired bootstrap, 10K iterations).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ReaLM-Retrieve Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Query → LRM → [Reasoning Chain Generation]                        │
│                        ↓                                            │
│               ①  Segment into Steps                                │
│                        ↓                                            │
│               ②  Compute RSUS Score                                │
│                  (α·U_verb + β·U_ent + γ·U_cons)                   │
│                        ↓                                            │
│               ③  Policy Decision π(s)                              │
│                        ↓                                            │
│               ④  Retrieve & Integrate (if needed)                  │
│                  • QueryGen formulates query                        │
│                  • ColBERTv2+PLAID retrieves docs                   │
│                  • Implicit decompression merges context            │
│                        ↓                                            │
│  Enhanced Chain → Final Answer                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**RSUS Components:**
- **U_verb**: Verbalized confidence from reasoning model self-assessment
- **U_ent**: Entity-based entropy measuring coverage ambiguity
- **U_cons**: Consistency across sampled continuations

---

## Installation

### Prerequisites
- Python 3.10 or 3.11
- CUDA 11.8+ (for GPU acceleration)
- 8× NVIDIA A100 80GB GPUs (for full training)
- 512GB RAM (for ColBERT indexing)

### Quick Start

```bash
# Clone repository
git clone https://github.com/bettyguo/realm-retrieve.git
cd realm-retrieve

# Install dependencies
make install-dev

# Download datasets and build indices
make data

# Train models
make train-all

# Evaluate
make eval
```

### Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install package
pip install -e ".[dev]"

# Download spaCy model
python -m spacy download en_core_web_sm

# Setup pre-commit hooks
pre-commit install
```

---

## Dataset Preparation

ReaLM-Retrieve evaluates on three multi-hop QA benchmarks:

| Dataset | Questions | Avg Hops | Corpus Size | Eval Split |
|---------|-----------|----------|-------------|------------|
| **MuSiQue** | 24,814 | 2-4 | 139K passages | test |
| **HotpotQA** | 90,564 | 2 | 5.2M passages | dev |
| **2WikiMHQA** | 192,606 | 2 | Wikipedia | test |

```bash
# Use make command
make data
```

Expected disk usage:
- Raw data: ~15GB
- Processed data: ~8GB
- ColBERT indices: ~45GB
- **Total: ~68GB**

---

## Training

### 1. Train Step Boundary Classifier

Segments reasoning chains into logical steps (94.2% F1, avg 127 tokens/step).

```bash
# Use make
make train-segmentation
```

**Training time:** ~4 hours on 8× A100 80GB

**Hyperparameters:**
- 3-layer transformer encoder, hidden_dim=256
- Learning rate: 5e-5
- Batch size: 32
- Epochs: 10

### 2. Train Retrieval Intervention Policy

REINFORCE-based policy learning for when to retrieve.

```bash
# Use make
make train-policy
```

**Training time:** ~12.5 days (2,400 GPU-hours on 8× A100)

**Hyperparameters:**
- REINFORCE with reward: F1(a_π, a*) - λ1·n_ret - λ2·t_latency
- Learning rate: 1e-4
- Curriculum learning: λ1 from 0.5 → 0.1
- Training steps: 50,000
- Batch size: 64

**RSUS weights** (optimized on validation):
- α (verbalized): 0.4
- β (entity entropy): 0.35
- γ (consistency): 0.25

---

## Evaluation

### Run Full Evaluation

```bash
# Evaluate on all three benchmarks
python evaluate.py configs/experiments/evaluate.yaml dataset=musique,hotpotqa,2wikimhqa

# Or use make
make eval
```

### Expected Results

**Table 1: Main Results on MuSiQue (DeepSeek-R1-32B)**

| Method | EM | F1 | Retrieval Calls | F1/Call Ratio |
|--------|----|----|-----------------|---------------|
| No Retrieval | 41.2 | 48.7 | 0.0 | - |
| Single RAG | 52.6 | 59.4 | 1.0 | 59.4 |
| IRCoT | 58.3 | 65.4 | 3.4 | 19.2 |
| FLARE | 55.1 | 62.3 | 2.8 | 22.3 |
| Self-RAG† | 54.8 | 61.9 | 2.1 | 29.5 |
| Search-R1 | 59.1 | 66.8 | 2.4 | 27.8 |
| **ReaLM-Retrieve** | **63.5** | **71.2** | **1.8** | **39.6** |

† Uses different base model (Llama-2-13B)

---

## Configuration

### Reasoning Models

```yaml
reasoning_models:
  deepseek_r1_32b:
    model_name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    backend: "vllm"
    max_tokens: 25000
    temperature: 0.7
    
  deepseek_r1_671b:
    model_name: "deepseek-ai/DeepSeek-R1"
    backend: "vllm"
    tensor_parallel_size: 8
    max_tokens: 25000
    
  openai_o1:
    model_name: "o1-preview"
    backend: "openai_api"
    api_key_env: "OPENAI_API_KEY"
    
  qwq_32b:
    model_name: "Qwen/QwQ-32B-Preview"
    backend: "vllm"
    max_tokens: 25000
```

### Retrieval Configuration

```yaml
retrieval:
  colbert:
    checkpoint: "colbert-ir/colbertv2.0"
    index_name: "musique.plaid"
    index_path: "data/indices/colbert"
    k_docs: 5
    max_passage_length: 256
    
  plaid:
    ncells: 2
    centroid_score_threshold: 0.5
    ndocs: 16384
```

### RSUS Configuration

```yaml
rsus:
  weights:
    alpha: 0.4      # Verbalized uncertainty
    beta: 0.35      # Entity entropy
    gamma: 0.25     # Consistency signal
  
  verbalized:
    prompt_template: "Given the reasoning so far, rate your confidence..."
    temperature: 0.3
    
  entity_entropy:
    ner_model: "en_core_web_sm"
    bm25_top_k: 100
    
  consistency:
    num_samples: 3
    temperature: 0.8
```

---

## Project Structure

```
realm-retrieve/
├── src/
│   ├── models/                  # Model implementations
│   │   ├── segmentation.py      # Step boundary classifier
│   │   ├── rsus.py              # RSUS computation
│   │   ├── policy.py            # Retrieval intervention policy
│   │   ├── retriever.py         # ColBERTv2 + PLAID wrapper
│   │   └── reasoning_model.py   # LRM API wrappers
│   └── evaluation/              # Metrics & evaluation
├── configs/                     # Hydra configurations
│   ├── models/                  # Model configs
│   ├── datasets/                # Dataset configs
│   └── experiments/             # Experiment configs
└── tests/                       # Unit & integration tests
```

---

## Baselines

ReaLM-Retrieve compares against:

1. **No Retrieval**: Closed-book reasoning model only
2. **Single RAG**: Standard retrieve-then-generate (1 retrieval before generation)
3. **IRCoT**: Fixed-interval retrieval after each sentence
4. **FLARE**: Token-probability-triggered retrieval
5. **Self-RAG**: Self-reflective retrieval with special tokens
6. **Search-R1**: RL-trained search decisions for reasoning models

---

## Evaluation Metrics

### Multi-Hop QA Metrics
- **EM (Exact Match)**: Percentage of perfectly matched answers
- **F1**: Token-level F1 score between prediction and ground truth
- **Sup-F1 (HotpotQA)**: F1 over supporting facts
- **Evi-F1 (2WikiMHQA)**: F1 over evidence sentences

### Retrieval Quality Metrics (pytrec_eval)
- **Recall@k**: Percentage of relevant docs in top-k
- **Precision@k**: Precision of top-k retrieved docs
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of first relevant doc
- **nDCG@k**: Normalized Discounted Cumulative Gain
- **Useful%**: Fraction of retrievals where ≥1 supporting fact appears in top-k

### Efficiency Metrics
- **Retrieval Calls**: Average number of retrieval invocations per question
- **E2E Latency**: End-to-end time from query to answer (seconds)
- **Per-Call Overhead**: Average time per retrieval call (seconds)
- **Reasoning Tokens**: Total tokens generated in reasoning chain
- **F1/Call Ratio**: Accuracy-efficiency trade-off metric

### Statistical Significance
All comparisons use **paired bootstrap resampling** (10,000 iterations) with **Bonferroni correction** for multiple comparisons. We report 95% confidence intervals and consider p < 0.05 statistically significant.

