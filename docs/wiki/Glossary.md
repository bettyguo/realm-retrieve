# Glossary

ReaLM-Retrieve borrows vocabulary from three communities (large reasoning
models, RAG/IR, and RL). This page disambiguates the most-overloaded terms.

---

### Adaptive retrieval

Any RAG variant that decides *when* to retrieve mid-generation, rather than
once before generation. Includes IRCoT, FLARE, Self-RAG, Search-R1, and
ReaLM-Retrieve. We use the term in the **policy-driven** sense — the
trigger is learned, not handcrafted.

### CoT — Chain of Thought

The sequence of tokens an LRM emits *before* its final answer, typically
delimited by `<think>...</think>` tags. In DeepSeek-R1 traces this can be
10K – 25K tokens. ReaLM-Retrieve operates **inside** the CoT.

### ColBERTv2 / PLAID

Late-interaction neural retriever. ColBERTv2 stores per-token contextual
embeddings; PLAID adds a centroid-based pruning layer that makes retrieval
~10× faster at ~equal recall. We use both by default.

### IRCoT

Baseline that retrieves after **every sentence** of a reasoning chain.
High recall, high cost — typically 3.4 retrievals per MuSiQue question.
ReaLM-Retrieve gets +5.8 F1 with 47 % fewer retrievals.

### LRM — Large Reasoning Model

A model trained (post-trained) to produce long chains of thought before
answering. Examples: **DeepSeek-R1**, **OpenAI o1**, **Qwen QwQ-32B-Preview**.
Distinct from a base LLM which answers in one shot.

### Multi-hop QA

A question whose answer requires composing information from ≥ 2 distinct
documents. Our benchmarks: **MuSiQue** (2-4 hops), **HotpotQA** (2 hops),
**2WikiMHQA** (2 hops).

### Policy (in this codebase)

Specifically the *retrieval-intervention* policy `π(retrieve | state) ∈ {0,1}`
in [`models/policy.py`](https://github.com/bettyguo/realm-retrieve/blob/main/src/realm_retrieve/models/policy.py).
Not to be confused with the LRM's own decoding "policy".

### REINFORCE

Policy-gradient method we use to train the retrieval-intervention policy.
With a moving baseline for variance reduction and curriculum-annealed cost
coefficient `λ₁: 0.5 → 0.1`. Why REINFORCE and not PPO/GRPO — see
[FAQ](FAQ#why-reinforce-and-not-ppo--grpo).

### Retrieval call

One invocation of the retriever, returning the top-k passages. In the paper
we report **retrieval calls per question** as the primary efficiency metric
since each call dominates end-to-end latency.

### RSUS — Reasoning Step Uncertainty Score

Per-step scalar in `[0, 1]` indicating how uncertain the LRM is about the
factual content of step `r_i`. Defined as

$$
\text{RSUS}(r_i) = \alpha \cdot U_{\text{verb}}(r_i) + \beta \cdot U_{\text{ent}}(r_i) + \gamma \cdot U_{\text{cons}}(r_i)
$$

with `(α, β, γ) = (0.40, 0.35, 0.25)` validated to sum to 1.

### Step (reasoning step)

A logical inference unit within a reasoning chain. Identified by the
boundary classifier (`models/segmentation.py`); avg 127 tokens. NOT a
sentence — sentences span ~23 tokens and cut across logical units.

### U_verb / U_ent / U_cons

The three RSUS components:

| Symbol     | Mnemonic         | What it captures                                          |
|------------|------------------|-----------------------------------------------------------|
| `U_verb`   | **Verbalized**   | Model's own self-rated confidence on a 0–100 scale.       |
| `U_ent`    | **Entity**       | Entropy over corpus coverage of entities named in the step. |
| `U_cons`   | **Consistency**  | 1 − agreement across `k=3` sampled continuations.          |

All in `[0, 1]`. Higher = more uncertain = more likely retrieval helps.

### WARP

The "Weight-Aggregated Retrieval with Persistence" inference optimisation we
apply on top of PLAID. Reduces per-call latency by ~3.2× by caching the
encoded query embedding across consecutive retrievals in the same chain.
