---
title:    "test(ci): add a CPU smoke test that imports every production module from a wheel"
labels:   ["enhancement", "ci", "good first issue", "testing"]
milestone: "v1.1"
state:    open
opened:   "2026-05-13"
---

## Why

Issue #3 fixed the worst of the heavy-import problem, but the only thing
defending against a regression is a developer remembering to `pip install -e .`
on a clean machine before pushing. We should encode that check in CI.

## Proposal

Add a new job to `.github/workflows/ci.yml`:

```yaml
import-smoke:
  name: CPU import smoke
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "3.11" }
    - run: |
        python -m pip install --upgrade pip build
        python -m build --wheel
        pip install dist/*.whl
        python -c "
        import importlib, sys
        mods = [
            'realm_retrieve',
            'realm_retrieve.cli',
            'realm_retrieve.toy',
            'realm_retrieve.models.segmentation',
            'realm_retrieve.models.rsus',
            'realm_retrieve.models.policy',
            'realm_retrieve.models.retriever',
            'realm_retrieve.models.reasoning_model',
            'realm_retrieve.evaluation.metrics',
        ]
        for m in mods:
            importlib.import_module(m)
            print(f'{m:55} OK')
        "
        realm-quickstart
```

The job installs **only** the base extras — no `vllm`, no `colbert`, no
`faiss-gpu`. Any future heavy top-level import will fail the build.

## Acceptance

- [ ] Job is wired into `ci.yml` and runs on `pull_request`.
- [ ] Job runs in < 60 s on the public runners.
- [ ] Removing the lazy-import in Issue #3 makes the job fail (regression
      coverage).

## Hints

Look at the existing `build` job for the wheel-building pattern. The
`realm-quickstart` console script is registered in `pyproject.toml`.
