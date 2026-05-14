---
title:     "chore(types): modernize type hints to PEP 604 (`X | None`, `list[T]`)"
labels:    ["good first issue", "chore", "documentation"]
milestone: "v1.1 (2026-06-11)"
state:     open
opened:    "2026-05-13"
---

## Background

The codebase predates our Python 3.10 minimum and still uses the legacy
`typing` aliases:

```python
from typing import Dict, List, Optional, Tuple

def foo(x: Optional[List[Dict[str, int]]]) -> Tuple[int, int]:
    ...
```

Python ≥ 3.10 prefers the built-in generics + `X | None`:

```python
def foo(x: list[dict[str, int]] | None) -> tuple[int, int]:
    ...
```

This is a stylistic improvement, not a behavioural change — but it makes the
codebase easier to skim and brings us in line with the newer files we just
shipped (`toy.py`, `cli.py`).

## What to do

1. Pick a module from
   [`src/realm_retrieve/models/`](../../../src/realm_retrieve/models) —
   `segmentation.py`, `rsus.py`, `policy.py`, `retriever.py`,
   `reasoning_model.py`. (One PR per module is fine.)
2. Replace `Optional[T]` → `T | None`, `List[T]` → `list[T]`,
   `Dict[K, V]` → `dict[K, V]`, `Tuple[A, B]` → `tuple[A, B]`.
3. Drop the now-unused imports from `typing`. Keep `TYPE_CHECKING`,
   `Protocol`, `Callable`, etc. where they appear.
4. Run `make lint format`. CI runs `ruff` and that will catch anything
   missed.

## Hints

- Add `from __future__ import annotations` at the very top — this means we
  don't need to keep wrapping forward references in strings.
- `ruff` rule `UP006` is the one that auto-fixes most of this:
  ```bash
  ruff check --select UP006,UP007 --fix src/realm_retrieve
  ```

## Acceptance

- [ ] No more `Optional` or generic `List` / `Dict` / `Tuple` in the touched
      module.
- [ ] `mypy src/realm_retrieve` stays green (run `make typecheck`).
- [ ] No imports left dangling.

This is a great way to get familiar with the codebase — happy to mentor in
the PR review.
