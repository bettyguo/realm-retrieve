---
title:    "bug(packaging): src/ layout drifts between src/realm_retrieve/ and flat src/"
labels:   ["bug", "imports", "tech-debt"]
milestone: "v1.1"
state:    open
opened:   "2026-05-14"
---

## Symptom

`pyproject.toml` declares

```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["realm_retrieve*"]
```

i.e. the package is expected to live at `src/realm_retrieve/`. But on at
least one developer's working tree the source files end up directly under
`src/__init__.py`, `src/cli.py`, etc. — without the `realm_retrieve/` parent
directory. In that state `pip install -e .` finds **zero** packages and the
codebase is silently broken.

This was observed during the v1.0.0 seeding work
([commit history Apr 29 – May 14](https://github.com/bettyguo/realm-retrieve/commits/main)).

## Likely cause

An auto-formatter / IDE extension is rewriting paths on save. The
`src/realm_retrieve.egg-info/SOURCES.txt` artifact still references
`src/realm_retrieve/*.py` (the canonical layout), confirming the drift
happens *after* the package was last built correctly.

## Fix

Choose one and commit to it:

- **Option A (recommended)**: enforce `src/realm_retrieve/` via a pre-commit
  hook that fails the commit if `src/__init__.py` exists.
- **Option B**: switch the packaging config to a flat-layout-tolerant
  setup using `package-dir = {"realm_retrieve" = "src"}` and an explicit
  `packages = [...]` list.

Option A is more conservative — fewer moving parts, no behaviour change for
consumers — so we prefer it.

## Acceptance

- [ ] A check that runs in CI (and ideally as a pre-commit hook) asserts
      `src/realm_retrieve/__init__.py` exists.
- [ ] `pip install -e .` from a fresh checkout finds at least 3 packages
      (`realm_retrieve`, `realm_retrieve.models`, `realm_retrieve.evaluation`).
- [ ] Add a smoke test in CI (overlaps with [#33](https://github.com/bettyguo/realm-retrieve/issues/33)).
