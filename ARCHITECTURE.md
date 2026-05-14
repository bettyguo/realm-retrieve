# Architecture (one-pager)

Long version lives in
[wiki / Architecture](https://github.com/bettyguo/realm-retrieve/wiki/Architecture)
and in **§4 of the SIGIR '26 paper**. This file is the elevator summary so
you don't have to leave the repo root.

```
question ──► LRM ──► reasoning chain (≈ 10K tokens)
                          │
                          ▼
        ① ReasoningStepSegmenter           — 3-layer transformer, 94.2 F1
                          │   steps_i ≈ 127 tokens
                          ▼
        ② RSUS = α·U_verb + β·U_ent + γ·U_cons   ∈ [0, 1]
                          │   (α, β, γ) = (0.40, 0.35, 0.25)
                          ▼
        ③ π(retrieve | state)              — REINFORCE policy, 4-layer transformer
                          │   binary action
                          ▼          ╔═══════════════════════╗
        ④ ColBERTv2 + PLAID ──── top-k ═► implicit decompression  ──► continue reasoning
                                          ╚═══════════════════════╝
                          ▼
                      final answer
```

## Component map

| Stage | Module | Lines | Trained? |
|-------|--------|------:|----------|
| ① Step segmentation | [`models/segmentation.py`](src/realm_retrieve/models/segmentation.py) | ~340 | 3-layer transformer, ~10 epochs |
| ② RSUS              | [`models/rsus.py`](src/realm_retrieve/models/rsus.py)                 | ~395 | weights (α,β,γ) optimised on val |
| ③ Policy            | [`models/policy.py`](src/realm_retrieve/models/policy.py)             | ~365 | REINFORCE, 50K steps             |
| ④ Retriever         | [`models/retriever.py`](src/realm_retrieve/models/retriever.py)       | ~125 | ColBERTv2 + PLAID (pretrained)   |
|     LRM adapter     | [`models/reasoning_model.py`](src/realm_retrieve/models/reasoning_model.py) | ~160 | none (wrapper)                   |

For each stage's design rationale (why this and not the obvious alternative),
see the **wiki Architecture page**.
