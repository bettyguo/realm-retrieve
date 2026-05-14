---
title:     "fix(rsus): validate that α + β + γ ≈ 1"
labels:    ["bug", "rsus"]
milestone: "v1.0"
state:     closed
opened:    "2026-05-11"
closed:    "2026-05-12"
closed_by: "ValueError raised at RSUSCalculator construction time"
---

## What's wrong

`RSUSCalculator.__init__` silently accepts arbitrary `(alpha, beta, gamma)`
weights. If a user copies the production values
(`0.4 / 0.35 / 0.25` = 1.0) and then re-tunes only one of them they end up
with a score that is **scaled but not bounded**, which breaks downstream
threshold comparisons in the policy.

## Reproduction

```python
calc = RSUSCalculator(model, ret, alpha=0.4, beta=0.4, gamma=0.4)   # sum = 1.2
score, _ = calc.compute_rsus(step, ctx)                              # may exceed 1.0
```

…then `policy(rsus=score > 0.65)` triggers far more often than intended.

## Fix

At construction time:

- Reject negative weights with `ValueError`.
- Reject `sum != 1` with absolute tolerance `1e-3` (lets validation-search
  results like `0.401 / 0.349 / 0.250` pass).

## Acceptance

- [x] Existing call sites with the paper's `(0.4, 0.35, 0.25)` weights still
      work.
- [x] `RSUSCalculator(..., alpha=0.4, beta=0.4, gamma=0.4)` raises
      `ValueError` with a useful message that names the bad sum.
- [x] Coverage added (will land in #2 / `tests/test_rsus_validation.py`).

## Side-quest

While we're here, also auto-download the spaCy NER model on first
construction so the offline failure mode is friendlier — see #3.
