---
title:    "refactor(types): migrate Optional[X] / List[X] to PEP 604 syntax"
labels:   ["enhancement", "good first issue", "help wanted", "tech-debt"]
milestone: "v1.1"
state:    open
opened:   "2026-05-13"
---

## Context

The codebase currently mixes pre-PEP-604 (`Optional[X]`, `List[X]`,
`Tuple[A, B]`) and modern (`X | None`, `list[X]`, `tuple[A, B]`) syntax. We
target Python 3.10+, so we can standardise on the modern spelling everywhere.

## Why this is a great first issue

- The change is mechanical (`ruff --select UP --fix` does most of it).
- It's spread over only six files, so the diff is reviewable end-to-end.
- You'll learn the project layout while doing it.

## Hints

```bash
# auto-fix everything ruff is confident about
ruff check --select UP006,UP007,UP035 --fix src tests

# review what remains
ruff check --select UP006,UP007,UP035 src tests
```

Files most affected:

- [src/realm_retrieve/models/segmentation.py](../../../src/realm_retrieve/models/segmentation.py)
- [src/realm_retrieve/models/rsus.py](../../../src/realm_retrieve/models/rsus.py)
- [src/realm_retrieve/models/policy.py](../../../src/realm_retrieve/models/policy.py)
- [src/realm_retrieve/models/retriever.py](../../../src/realm_retrieve/models/retriever.py)
- [src/realm_retrieve/models/reasoning_model.py](../../../src/realm_retrieve/models/reasoning_model.py)
- [src/realm_retrieve/evaluation/metrics.py](../../../src/realm_retrieve/evaluation/metrics.py)

Don't forget to add `from __future__ import annotations` if you need to use
the new syntax at module scope without breaking on 3.9 (we won't, but the
import is harmless).

## Acceptance

- [ ] `ruff check --select UP006,UP007,UP035 src tests` → clean.
- [ ] `make test` passes.
- [ ] `make typecheck` passes.

## Reward

You get top billing in `CHANGELOG.md` for the v1.1 release. Comment to claim.
