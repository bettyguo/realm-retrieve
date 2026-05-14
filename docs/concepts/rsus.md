# RSUS — Reasoning Step Uncertainty Score

A single scalar in `[0, 1]` per reasoning step. High → likely a knowledge gap →
worth retrieving.

```
RSUS(r_i) = α · U_verb(r_i) + β · U_ent(r_i) + γ · U_cons(r_i)
```

| Signal     | Captures                                                                 | Cost     |
|------------|--------------------------------------------------------------------------|----------|
| `U_verb`   | Verbalised self-assessment — the model rates its own confidence (0–100). | 1 short generation |
| `U_ent`    | Entity-coverage entropy — sparse retrieval distribution = ambiguity.     | 1 BM25 call per entity |
| `U_cons`   | Consistency across `k=3` sampled continuations.                          | k continuations |

We learned `α = 0.40`, `β = 0.35`, `γ = 0.25` on the MuSiQue validation split.
The consistency signal is the most expensive, so it is only computed for
*critical* steps (heuristic: discourse markers like "therefore" / "thus" / …).

See [`realm_retrieve.models.rsus.RSUSCalculator`](../reference/api.md).
