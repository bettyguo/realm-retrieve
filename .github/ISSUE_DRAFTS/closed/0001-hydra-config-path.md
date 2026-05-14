---
title:    "fix(evaluate): Hydra config_path points to a non-existent directory"
labels:   ["bug", "config", "good first issue"]
milestone: "v1.0"
state:    closed
assignees: ["bettyguo"]
opened:   "2026-04-20"
closed:   "2026-04-22"
closed_by: "fix(evaluate): move evaluate.yaml under configs/experiments (#1)"
---

## Reproduction

```bash
python evaluate.py dataset=musique
```

```
hydra.errors.MissingConfigException: Cannot find primary config 'evaluate'.
Searched in:
  ${HOME}/realm-retrieve/configs/experiments
```

## Root cause

`evaluate.py` declares

```python
@hydra.main(version_base=None, config_path="configs/experiments", config_name="evaluate")
```

but the actual file lives at `configs/evaluate.yaml`. The `configs/experiments/`
directory does not exist in the repo at all, even though the README references
it and the two `train_*.py` scripts assume the same layout.

## Fix

Adopt the layout the README already describes:

```
configs/experiments/
├── evaluate.yaml
├── train_segmentation.yaml
└── train_policy.yaml
```

- Move `configs/evaluate.yaml` → `configs/experiments/evaluate.yaml`.
- Add the two training configs that match the hyper-parameters reported in §4
  of the paper.
- Leave the `@hydra.main(...)` decorators untouched so external scripts that
  already use them keep working.

## Verification

```bash
python evaluate.py dataset=musique max_examples=5      # no MissingConfigException
python train_segmentation.py                            # config resolves
python train_policy.py                                  # config resolves
```
