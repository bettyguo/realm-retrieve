---
title:     "feat(configs): ship the `configs/experiments/*.yaml` referenced by trainers"
labels:    ["enhancement", "config", "good first issue"]
milestone: "v1.0"
state:     closed
opened:    "2026-05-06"
closed:    "2026-05-08"
closed_by: "configs/experiments/train_{policy,segmentation}.yaml added"
---

## Context

`train_policy.py` and `train_segmentation.py` both reference Hydra configs
that don't ship with the repo, e.g.:

```python
@hydra.main(version_base=None, config_path="configs/experiments", config_name="train_policy")
```

So a new user cloning the repo and running `make train-policy` gets an
unhelpful "config not found" error. The README documents all the
hyperparameters but does not include the YAML files themselves.

## What to ship

| File | Notes |
|------|-------|
| `configs/experiments/train_segmentation.yaml` | base model, hidden_dim, lr 5e-5, 10 epochs |
| `configs/experiments/train_policy.yaml`       | curriculum λ₁ 0.5 → 0.1, 50k steps, lr 1e-4 |
| `configs/experiments/evaluate.yaml`           | moved from `configs/evaluate.yaml` |

Each YAML should declare `defaults: [_self_]` and pull seeds from a single
top-level `seed:` field for reproducibility.

## Acceptance

- [x] Trainers can be launched with no CLI overrides.
- [x] Values match those documented in the README (lr, epochs, dropout, …).
- [x] `make ci` still passes.

## Out of scope

- Per-dataset configs under `configs/datasets/` — left for #8 (Retriever
  protocol) since the dataset / index pair is part of the swap.
