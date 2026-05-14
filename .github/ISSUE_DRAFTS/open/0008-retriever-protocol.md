---
title:    "feat(retriever): extract a Retriever Protocol so users can plug in BM25 / SPLADE / custom backends"
labels:   ["enhancement", "help wanted", "architecture"]
milestone: "v1.2"
state:    open
opened:   "2026-05-13"
---

## Motivation

Right now the pipeline is hard-coded to `ColBERTRetriever`. Multiple folks have
asked on Discussions whether they can swap in:

- **BM25** (Pyserini / Elastic) — for a low-cost CPU baseline.
- **SPLADE** — for a sparse-dense ablation.
- **Their own internal retriever** — common in industry deployments.

Today this means subclassing `ColBERTRetriever` and overriding `.retrieve()`,
which couples them to ColBERT's runtime imports.

## Proposal

Define a tiny `typing.Protocol`:

```python
# src/realm_retrieve/models/retriever_protocol.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        k: int | None = None,
        return_scores: bool = False,
    ) -> list[dict]: ...

    def get_corpus_size(self) -> int: ...
```

- Update `RSUSCalculator.__init__` and the `ReaLMRetrieveSystem` constructor
  to type-annotate against `Retriever` rather than `ColBERTRetriever`.
- Add a thin `BM25Retriever` in `realm_retrieve.models.bm25` (the toy code in
  `realm_retrieve.toy.ToyRetriever` is 90% of this already).
- Add a Hydra plugin point: `retrieval.kind: colbert | bm25 | custom` with a
  `_target_:` Hydra instantiation for the `custom` case.

## Acceptance

- [ ] Existing `ColBERTRetriever` satisfies the protocol with **zero**
      changes.
- [ ] `BM25Retriever` lives in `src/realm_retrieve/models/bm25.py`.
- [ ] `tests/test_retriever_protocol.py` verifies both implementations
      satisfy the protocol at runtime.
- [ ] README's *Plug-in Retriever protocol* roadmap bullet links here.

## Out of scope

- Re-running the paper's experiments with non-ColBERT retrievers. We'll do
  that as a follow-up research issue (`#TBD`) once the plumbing exists.
