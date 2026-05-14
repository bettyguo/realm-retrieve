---
title:     "test(ci): add a CPU smoke test that wires the full pipeline from a wheel"
labels:    ["good first issue", "testing", "ci"]
milestone: "v1.1 (2026-06-11)"
state:     open
opened:    "2026-05-13"
---

## Why

Today CI runs unit tests against the source tree. That tests the code but
doesn't test the **packaging contract** — i.e. that someone running

```bash
pip install realm-retrieve
```

…gets a working set of importable classes. Issue #3 fixed the imports; we
should pin that behaviour with a test so it can't silently regress.

## What to build

A new pytest marked `@pytest.mark.integration` that:

1. Builds a wheel (`python -m build`).
2. Installs it into a clean `pip` venv (use `tmp_path` + `subprocess`).
3. Runs a tiny Python snippet that:
   - imports `ReasoningModelWrapper`, `ColBERTRetriever`, `RSUSCalculator`
     **without** actually invoking them,
   - imports and runs `realm_retrieve.cli._cmd_quickstart`.
4. Asserts:
   - exit code 0,
   - `vllm`, `colbert`, `spacy` are not in `sys.modules` of the subprocess.

CI is already split into a fast lane and an `integration` lane — add the new
test there.

## Hints

- `tests/test_imports.py` already does a single-process version of this. The
  challenge is the cross-process / clean-env piece.
- A minimal wheel can be built with `python -m build --wheel --outdir <dir>`.
- For the subprocess imports, dump `list(sys.modules)` to a file and assert
  on it from the parent process.

## Acceptance

- [ ] New test marked `@pytest.mark.integration`.
- [ ] CI lane "integration" runs it on `ubuntu-latest` and `windows-latest`.
- [ ] Test fails if any of the three forbidden modules leak.
