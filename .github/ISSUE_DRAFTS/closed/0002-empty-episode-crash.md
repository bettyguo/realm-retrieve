---
title:    "fix(policy): REINFORCETrainer.train_step crashes on empty episodes"
labels:   ["bug", "training", "rl"]
milestone: "v1.0"
state:    closed
assignees: ["bettyguo"]
opened:   "2026-04-25"
closed:   "2026-04-28"
closed_by: "fix(policy): short-circuit train_step for zero-length rollouts (#2)"
---

## Reproduction

Run the bundled training entry-point. The placeholder rollout passes empty
lists for `states` / `actions`:

```python
metrics = trainer.train_step(
    states=[],          # ← will crash inside torch.stack
    actions=[],
    f1_score=0.7,
    num_retrievals=2,
    total_latency=0.5,
)
```

```
RuntimeError: stack expects a non-empty TensorList
```

## Impact

- Every fresh checkout dies on the very first training step.
- More importantly, valid rollouts that legitimately produce no policy
  decisions (e.g. the LRM answered without entering the reasoning loop) raise
  the same exception in production.

## Fix

Short-circuit at the top of `REINFORCETrainer.train_step`:

- Still update the baseline (we have a scalar reward).
- Skip the policy-gradient and entropy terms.
- Return a metrics dict with `skipped_empty_episode: True` so trainers can log
  the rate of empty rollouts.

## Tests

A regression test in [`tests/test_policy_trainer.py`](../../../tests/test_policy_trainer.py)
exercises the empty-episode path and asserts the metrics dict matches.
