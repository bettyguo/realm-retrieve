# Architecture

A guided tour of the four-stage ReaLM-Retrieve pipeline. Cross-references the
SIGIR '26 paper §4 and the source modules under
[`src/realm_retrieve/models/`](https://github.com/bettyguo/realm-retrieve/tree/main/src/realm_retrieve/models).

---

## ① Reasoning step segmentation

Large reasoning models emit *one* long stream of tokens. To reason about
retrieval at all we first need to chunk that stream into **logical steps** —
units of inference that resolve a single sub-claim.

- **Where**: [`models/segmentation.py`](https://github.com/bettyguo/realm-retrieve/blob/main/src/realm_retrieve/models/segmentation.py)
- **Model**: 3-layer transformer boundary classifier, hidden 256, 4 heads.
- **Trained on**: 2,847 human-annotated DeepSeek-R1 traces.
- **Performance**: 94.2 F1 on a held-out test set (N = 412 traces).
- **Step length** (post-segmentation): avg 127 tokens (vs. 23 tokens / sentence).

### Why not sentence-level (IRCoT)?

Sentences cut across logical units in long CoT. Empirically, sentence-level
RSUS triggers ~3.4 retrieval calls per question; step-level triggers ~1.8 with
*higher* F1.

### Why not paragraph-level?

Paragraphs over-coalesce in reasoning chains where the model "switches gears"
mid-paragraph (e.g. "Wait — let me reconsider..."). Discourse-marker training
data captures these correctly.

---

## ② Reasoning Step Uncertainty Score (RSUS)

Per-step uncertainty signal in `[0, 1]`. Three orthogonal components,
linearly combined with weights optimised on a held-out validation split:

$$
\text{RSUS}(r_i) = \alpha \cdot U_{\text{verb}}(r_i) + \beta \cdot U_{\text{ent}}(r_i) + \gamma \cdot U_{\text{cons}}(r_i)
$$

With (α, β, γ) = (**0.40, 0.35, 0.25**).

| Component | What it measures | Cost | Best for |
|-----------|-----------------|------|----------|
| `U_verb`  | Model self-assessment ("rate confidence 0–100"). | 1 extra forward pass. | Models that are well-calibrated by RLHF. |
| `U_ent`   | Entropy of corpus coverage over entities named in the step. | 1 retrieval per entity (cached). | Steps that *introduce* novel entities. |
| `U_cons`  | Disagreement across `k=3` sampled continuations. | 3× sampling cost (only on "critical" steps). | Branch points where the model is exploring. |

See [`models/rsus.py`](https://github.com/bettyguo/realm-retrieve/blob/main/src/realm_retrieve/models/rsus.py)
for the implementation. The constructor validates `α + β + γ ≈ 1` after
[issue #5](https://github.com/bettyguo/realm-retrieve/issues/31).

---

## ③ Retrieval intervention policy

A learned binary classifier `π(retrieve | state) ∈ {0, 1}`. **Not** simply
"threshold RSUS at 0.65" — the policy also has access to:

- Embedding of the original query.
- Embedding of the current step.
- History of past retrievals in this trace.
- Position (step number / estimated total).

so it can learn things like "I already retrieved twice on this question; the
third retrieval is unlikely to help even if RSUS is high".

- **Where**: [`models/policy.py`](https://github.com/bettyguo/realm-retrieve/blob/main/src/realm_retrieve/models/policy.py)
- **Model**: 4-layer transformer (hidden 512, 8 heads), single Bernoulli head.
- **Trained with**: REINFORCE + curriculum-annealed cost coefficient
  λ₁: 0.5 → 0.1 over 50K steps. Baseline subtraction for variance reduction.
- **Reward**: `R = F1(â, a*) − λ₁ · n_ret − λ₂ · t_latency`.

### Empty-rollout safety

After [issue #2](https://github.com/bettyguo/realm-retrieve/issues/29), the
trainer short-circuits on zero-length rollouts (still updates the baseline,
skips backprop, returns `skipped_empty_episode: True`).

---

## ④ Retriever & context fusion

[`models/retriever.py`](https://github.com/bettyguo/realm-retrieve/blob/main/src/realm_retrieve/models/retriever.py)
wraps **ColBERTv2** with the **PLAID/WARP** late-interaction index. Defaults:

- `query_maxlen=256`, `doc_maxlen=256`
- `ncells=2`, `centroid_score_threshold=0.5`, `ndocs=16384`
- top-k = 5 passages per call

The retrieved passages are spliced into the reasoning context with an
**implicit decompression** template that compresses 5 passages × 256 tokens
into ~600 tokens of inline evidence (vs. the naïve 1,280-token concatenation).
This is where most of the 3.2× per-call latency reduction comes from.

A pluggable `Retriever` protocol (BM25, SPLADE, custom) is tracked by
[issue #8](https://github.com/bettyguo/realm-retrieve/issues/34).

---

## Five-line mental model

```python
chain = lrm.generate(question)                                 # the LRM
steps = segmenter.segment(chain)                               # ①
for i, step in enumerate(steps):
    score, components = rsus.compute_rsus(step, context=...)   # ②
    if policy.decide(score, components, history=...):           # ③
        docs = retriever.retrieve(query=question + step)        # ④
        chain = lrm.continue_with(chain, docs)                  # context fusion
answer = extract_answer(chain)
```

The [`examples/quickstart.py`](https://github.com/bettyguo/realm-retrieve/blob/main/examples/quickstart.py)
walks through this exact flow on a 12-document toy corpus, in <2 s on CPU.
