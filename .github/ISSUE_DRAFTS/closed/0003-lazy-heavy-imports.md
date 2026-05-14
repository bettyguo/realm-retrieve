---
title:    "fix(models): heavy backends (vllm / colbert / spacy) must be lazy-imported"
labels:   ["bug", "imports", "dx"]
milestone: "v1.0"
state:    closed
assignees: ["bettyguo"]
opened:   "2026-05-02"
closed:   "2026-05-05"
closed_by: "fix(models): lazy-load vllm, colbert and spacy at first use (#3)"
---

## Symptom

On a CPU-only laptop (or any environment without CUDA + faiss-gpu), simply
**importing** the package — never mind calling anything — fails:

```python
>>> from realm_retrieve.models.reasoning_model import ReasoningModelWrapper
ImportError: cannot import name 'LLM' from 'vllm' (... no module named 'vllm')
```

```python
>>> from realm_retrieve.models.retriever import ColBERTRetriever
ModuleNotFoundError: No module named 'colbert'
```

This blocks:

- the **quickstart demo** (CPU only by design),
- IDE type-checking and docstring rendering,
- unit tests that monkey-patch the wrappers,
- API-only users who never want to install vLLM in the first place.

## Fix

Move `vllm`, `openai`, `colbert*`, and `spacy` imports **inside** the methods
that actually need them. Keep `torch` / `numpy` / `transformers` at the top —
they're cheap and always installed by the base extras.

When `spacy.load("en_core_web_sm")` fails because the model isn't downloaded,
auto-fetch it once (`spacy.cli.download`) rather than dying with the cryptic
`OSError [E050]` that ships with spaCy.

## Out of scope

We're *not* changing the public API or constructor signatures — only the
import discipline. Existing call sites continue to work unmodified.
