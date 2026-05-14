---
title:    "feat(configs): ship the train_segmentation / train_policy Hydra configs"
labels:   ["enhancement", "config", "reproducibility"]
milestone: "v1.0"
state:    closed
assignees: ["bettyguo"]
opened:   "2026-05-06"
closed:   "2026-05-08"
closed_by: "feat(configs): add experiment configs referenced by trainers (#4)"
---

## Problem

Both training scripts decorate `main` with

```python
@hydra.main(..., config_path="configs/experiments", config_name="train_segmentation")
@hydra.main(..., config_path="configs/experiments", config_name="train_policy")
```

but the corresponding YAML files are missing from the repo. Anyone running
`make train-segmentation` or `make train-policy` hits the same Hydra
`MissingConfigException` as Issue #1.

## Fix

Ship the configs that exactly match the hyper-parameters reported in the
paper:

- `configs/experiments/train_segmentation.yaml`
  – 3-layer transformer, hidden 256, batch 32, LR 5e-5, 10 epochs.
- `configs/experiments/train_policy.yaml`
  – 4-layer transformer policy, REINFORCE w/ curriculum λ₁ 0.5→0.1,
    50K steps, baseline momentum 0.95, entropy coefficient 0.01.

Both seed with `0xRE4L` so a fresh checkout reproduces the paper numbers
deterministically given the same data shards.

## Future

Tracked separately: a Hydra **sweep** config for the ablation grid in §6.
