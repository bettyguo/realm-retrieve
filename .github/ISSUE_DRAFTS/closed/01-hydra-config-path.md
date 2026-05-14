---
title:     "fix(evaluate): Hydra `config_path` points at non-existent directory"
labels:    ["bug", "config"]
milestone: "v1.0"
state:     closed
opened:    "2026-04-22"
closed:    "2026-04-22"
closed_by: "configs/experiments/* committed in this release"
---

## What's wrong

`evaluate.py:197` and both training entry points pass
`config_path="configs/experiments"` to `@hydra.main`, but the directory does
not exist and the only Hydra config in the tree is `configs/evaluate.yaml`.
Running any of the entry points blows up immediately:

```
Primary config directory not found.
Check that the config path 'configs/experiments' is correct.
```

The README and `Makefile` both reference `configs/experiments/*.yaml`, so the
right fix is to **create that directory**, not patch the entry points.

## Reproduction

```bash
python evaluate.py dataset=musique
```

## Fix

- Move `configs/evaluate.yaml` → `configs/experiments/evaluate.yaml`.
- Add the matching `configs/experiments/train_segmentation.yaml` and
  `configs/experiments/train_policy.yaml` referenced by the trainers.
- Ensure both new YAMLs round-trip through Hydra with `_self_` in `defaults`.

## Acceptance

- [x] `python evaluate.py dataset=musique` no longer errors during config
      resolution.
- [x] `make train-segmentation` and `make train-policy` start cleanly.
- [x] `pytest tests/` stays green.
