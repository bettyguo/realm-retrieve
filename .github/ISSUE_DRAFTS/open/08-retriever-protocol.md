---
title:     "feat(retriever): extract a `Retriever` Protocol so users can plug in BM25 / SPLADE / custom backends"
labels:    ["enhancement", "help wanted", "design"]
milestone: "v1.2 (2026-07-09)"
state:     open
opened:    "2026-05-13"
---

## Motivation

Right now the system assumes a `ColBERTRetriever`. Several early adopters
have reached out asking how to swap in:

- **BM25 / Pyserini / Anserini** — already trained on their corpus.
- **SPLADE / SPLADE++** — sparse-dense hybrid.
- A bespoke **enterprise vector DB** (Qdrant, Weaviate, Vespa, custom).

Today this requires monkey-patching `evaluate.py`. We should publish a
formal contract so adapters can live outside the repo.

## Proposed surface

```python
# src/realm_retrieve/models/protocol.py
from typing import Protocol, TypedDict


class RetrievedPassage(TypedDict, total=False):
    passage_id: str
    text:       str
    rank:       int
    score:      float


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        k: int = 5,
        return_scores: bool = False,
    ) -> list[RetrievedPassage]: ...

    def get_corpus_size(self) -> int: ...
```

Then `ColBERTRetriever` and `ToyRetriever` both keep working unchanged
(structural subtyping), and we can document the contract in
[`docs/concepts/retrieval.md`](../../../docs/concepts/retrieval.md).

## Subtasks

- [ ] Add `protocol.py` with the `Retriever` `Protocol` and a `RetrievedPassage`
      `TypedDict`.
- [ ] Re-export `Retriever` from `realm_retrieve.__init__` (lazily).
- [ ] Write an example adapter for **BM25 via rank-bm25** under
      `examples/adapters/bm25_rankbm25.py`.
- [ ] Update `evaluate.py` to take an adapter path through Hydra
      (`retriever._target_:` string) instead of hard-coding `ColBERTRetriever`.
- [ ] Document migration in `docs/concepts/retrieval.md`.

## Acceptance

- [ ] `make eval DATASET=musique retriever=bm25` runs the BM25 adapter
      end-to-end on a small toy index.
- [ ] `pytest -m integration` covers two adapters (toy BM25 + ColBERT mock).
- [ ] mkdocs docs include a "Bring your own retriever" tutorial.

If you'd like to own this, drop a comment — it's a great chance to shape an
extension point that several teams want.
