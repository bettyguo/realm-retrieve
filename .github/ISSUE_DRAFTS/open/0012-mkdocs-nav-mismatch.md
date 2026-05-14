---
title:    "docs(mkdocs): nav references files that don't exist (broken `mkdocs build`)"
labels:   ["bug", "documentation", "good first issue"]
milestone: "v1.1"
state:    open
opened:   "2026-05-14"
---

## Symptom

`mkdocs.yml` declares a navigation tree like:

```yaml
nav:
  - Home: index.md
  - Getting started:
      - Installation: getting-started/installation.md
      - CPU quickstart: getting-started/quickstart.md
  ...
```

…but the actual docs tree is:

```
docs/
├── index.md
├── install.md
├── quickstart.md
├── inference.md
├── concepts/
├── reference/
├── contributing.md
└── changelog.md
```

So `mkdocs build` would fail (or silently emit warnings about missing pages)
and `make docs-serve` will not render half the navigation.

## Fix

Two paths:

- **Reorganise the files** to match the nav structure declared in
  `mkdocs.yml`. (More work; canonical layout per mkdocs-material conventions.)
- **Rewrite the nav** to match the flat layout that's actually on disk.
  (Faster; pragmatic.)

Either is fine for v1.1. Pick one and execute end-to-end so a fresh
`make docs-serve` produces zero warnings.

## Hints

```bash
make docs-serve      # locally; warnings appear on stderr
make docs            # one-shot build, stricter
```

The `git-revision-date-localized` plugin needs full git history, so make
sure `actions/checkout@v4` in `.github/workflows/docs.yml` keeps
`fetch-depth: 0`.

## Acceptance

- [ ] `mkdocs build` exits 0 with **zero warnings** on a clean checkout.
- [ ] Every entry in the nav resolves to a real file.
- [ ] Every `.md` under `docs/` is reachable from the nav (or explicitly
      excluded via `not_in_nav` if appropriate).
