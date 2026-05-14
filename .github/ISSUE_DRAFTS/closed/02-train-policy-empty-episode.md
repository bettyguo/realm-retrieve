---
title:     "fix(policy): `REINFORCETrainer.train_step` crashes on empty episode"
labels:    ["bug", "training"]
milestone: "v1.0"
state:     closed
opened:    "2026-04-28"
closed:    "2026-04-28"
closed_by: "Guard added in src/realm_retrieve/models/policy.py:288"
---

## What's wrong

The placeholder training loop in
[`train_policy.py:73-87`](../../../train_policy.py) passes `states=[]` and
`actions=[]` for every step before the real environment is wired up. Inside
`REINFORCETrainer.train_step` this hits:

```python
log_probs = torch.stack([...])           # ← empty list
policy_loss = -(log_probs * advantage).mean()
```

`torch.stack` on an empty list raises `RuntimeError: stack expects a non-empty
TensorList`, and the run dies before the first checkpoint.

## Fix

Short-circuit empty episodes — still update the running baseline (so warmup
batches contribute their reward signal), bump `training_step`, and return a
metrics dict with `loss=0.0` and a `skipped_empty_episode=True` flag so
downstream loggers can filter them.

## Acceptance

- [x] `trainer.train_step([], [], f1_score=0.7, num_retrievals=2, total_latency=0.5)`
      returns a metrics dict without raising.
- [x] Real (non-empty) episodes are unaffected — same loss, same gradients.
- [x] The skipped-episode flag is wired through the wandb log lines so we can
      see when warmup is over.

## Notes for v1.1

- Once the real environment loop is in place, the placeholder should be deleted
  outright. Tracked separately in #7.
