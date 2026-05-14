---
title:     "fix(imports): heavy submodule imports break CPU-only consumers"
labels:    ["bug", "packaging"]
milestone: "v1.0"
state:     closed
opened:    "2026-05-02"
closed:    "2026-05-05"
closed_by: "Lazy imports in reasoning_model.py / retriever.py / rsus.py"
---

## What's wrong

Three submodules pull heavy / GPU-leaning packages at module top:

| File                           | Heavy import         |
|--------------------------------|----------------------|
| `models/reasoning_model.py`    | `vllm`, `openai`     |
| `models/retriever.py`          | `colbert.*`          |
| `models/rsus.py`               | `spacy`              |

This means **even a CPU-only consumer** doing

```python
from realm_retrieve.models.reasoning_model import ReasoningModelWrapper
```

…crashes with `ModuleNotFoundError: No module named 'vllm'`, which defeats the
purpose of the new top-level lazy-import shim (#0 / `__getattr__`).

## Reproduction

```bash
pip install -e . --no-deps              # minimal env, no vllm/colbert/spacy
python -c "from realm_retrieve.models.retriever import ColBERTRetriever"
```

## Fix

- Move `from vllm import LLM, SamplingParams` into `VLLMReasoningModel.__init__`
  and `.generate`.
- Move `import openai` into `OpenAIReasoningModel.__init__`.
- Move `from colbert import Searcher; from colbert.infra import …` into
  `ColBERTRetriever.__init__`.
- Move `import spacy` into `RSUSCalculator.__init__`; auto-download
  `en_core_web_sm` if missing (closes #4 as a side effect).

## Acceptance

- [x] Adding a `tests/test_imports.py::test_top_level_import_does_not_pull_heavy_deps`
      that loads every wrapper without `vllm` / `colbert` / `spacy` / `openai`
      ending up in `sys.modules`.
- [x] CI fast lane installs `torch` (CPU wheel) + light deps only and still
      passes.
- [x] The Docker quickstart image (`realm-retrieve:latest`) builds without
      vLLM or ColBERT.
