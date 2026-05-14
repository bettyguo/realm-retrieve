---
title:    "refactor(types): turn on strict mypy mode in CI"
labels:   ["enhancement", "tech-debt", "good first issue"]
milestone: "v1.2"
state:    open
opened:   "2026-05-14"
---

## Context

`pyproject.toml` currently runs mypy with deliberately lenient settings:

```toml
[tool.mypy]
python_version = "3.10"
warn_unused_configs = true
disallow_untyped_defs = false       # ← lenient
ignore_missing_imports = true       # ← lenient
```

This was fine for the rush to camera-ready, but it lets a lot of `Any`-typed
code through. We'd like to tighten this incrementally toward `--strict`.

## Proposed phased approach

Don't flip everything at once — it'll create a 200-error wall that nobody
fixes. Instead:

### Step 1 (v1.2)

Flip `disallow_untyped_defs = true` on **one module at a time**, starting
with the most-touched-by-newcomers:

1. `src/realm_retrieve/toy.py` (lowest dep, biggest reach for newcomers)
2. `src/realm_retrieve/evaluation/metrics.py`
3. `src/realm_retrieve/cli.py`
4. `src/realm_retrieve/models/segmentation.py`
5. `src/realm_retrieve/models/rsus.py`
6. `src/realm_retrieve/models/policy.py`
7. `src/realm_retrieve/models/retriever.py`
8. `src/realm_retrieve/models/reasoning_model.py`

Use mypy per-module overrides:

```toml
[[tool.mypy.overrides]]
module = "realm_retrieve.toy"
disallow_untyped_defs = true
```

### Step 2 (v1.3+)

Once all modules pass `disallow_untyped_defs`, flip the global flag and
remove the overrides.

### Step 3 (v2.0)

Add `--strict-equality`, `--no-implicit-reexport`, etc., one at a time.

## Acceptance (this issue scope = Step 1)

- [ ] All 8 modules above have `disallow_untyped_defs = true` overrides.
- [ ] CI `make typecheck` is green.
- [ ] PR description lists the most common type-hint patterns added so the
      next contributor can pattern-match.

## Why good first issue

Adding type hints is mechanical, **doesn't change behaviour**, and forces
you to read every module — a great way to learn the codebase. Pick any one
module from the list and send a PR; we'll merge each independently.
