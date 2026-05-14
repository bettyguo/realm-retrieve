---
title:    "fix(rsus): silently accept α + β + γ ≠ 1 produces miscalibrated scores"
labels:   ["bug", "rsus", "good first issue"]
milestone: "v1.0"
state:    closed
assignees: ["bettyguo"]
opened:   "2026-05-10"
closed:   "2026-05-12"
closed_by: "fix(rsus): validate weights sum to 1 at construction (#5)"
---

## Symptom

A user typo'd `alpha=0.4, beta=0.35, gamma=0.35` (sum = 1.10) in their YAML and
spent half a day debugging an unexpected F1 drop on MuSiQue. The constructor
accepted the bad config silently, RSUS scores ended up scaled outside `[0, 1]`,
and the downstream retrieval threshold fired far more often than intended.

## Fix

`RSUSCalculator.__init__` now hard-fails fast:

```python
if not math.isclose(alpha + beta + gamma, 1.0, abs_tol=1e-3):
    raise ValueError(
        f"RSUS weights must sum to 1; got alpha={alpha}, beta={beta}, "
        f"gamma={gamma} (sum={total:.6f})"
    )
```

Also rejects negative weights (which would invert the signal sign).

## Why a hard error and not a warning

The paper's derivation (§3.2, Eq. 4) assumes `α + β + γ = 1` so that RSUS lives
on `[0, 1]` and the policy threshold has a fixed semantic. A warning would be
quietly ignored in long training runs; a hard error surfaces the mistake the
moment the user wires up the config.

## Tests

Unit test [`tests/test_rsus.py::test_weight_validation`](../../../tests/test_rsus.py)
covers both the sum constraint and the non-negativity constraint.
