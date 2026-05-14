# Retrieval Intervention Policy

A small (≈ 4 M parameter) transformer that maps `state → P(retrieve)`.

## Input state

| Field                  | Shape       | Meaning                                              |
|------------------------|-------------|------------------------------------------------------|
| `query_embedding`      | `[d]`       | Sentence-transformer encoding of the original query. |
| `current_step_embedding` | `[d]`     | Embedding of the current reasoning step.             |
| `rsus_features`        | `[3]`       | Raw RSUS components (verb, ent, cons).               |
| `retrieval_history`    | `[10]`      | Bag of features summarising prior retrieval events.  |
| `step_number`          | `int`       | Positional embedding (0–199).                        |

## Reward

```
R  =  F1(answer_π, answer*)  −  λ₁ · n_retrievals  −  λ₂ · t_latency
```

- `λ₁` follows a **curriculum**: starts at 0.5 (encourage parsimony), anneals
  linearly to 0.1 over 50 K steps (let the policy use evidence when it
  actually helps).
- `λ₂` is fixed at `1e-2` and serves mainly as a tie-breaker.

## Training

`make train-policy` — 50 K REINFORCE steps, baseline = exponential moving
average of returns, entropy regularisation `1e-2`.

See [`realm_retrieve.models.policy.REINFORCETrainer`](../reference/api.md).
