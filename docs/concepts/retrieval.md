# Retrieval

The default backend is **ColBERTv2 + PLAID** for production and a pure-Python
**BM25** for the CPU quickstart.

## ColBERTv2 + PLAID

```python
from realm_retrieve import ColBERTRetriever

retriever = ColBERTRetriever(
    index_path="data/indices/colbert/musique.plaid",
    checkpoint="colbert-ir/colbertv2.0",
    k=5,
)
docs = retriever.retrieve("What year did the Berlin Wall fall?", k=5)
```

### Index build

```bash
make data DATASET=musique
```

The index lives under `data/indices/colbert/<dataset>.plaid` (~ 15 GB
per dataset).

## BM25 toy

```python
from realm_retrieve.toy import ToyRetriever, demo_corpus

bm25 = ToyRetriever(demo_corpus(), k=3)
docs = bm25.retrieve("Olympics Beijing")
```

## Bring your own

Implement the [`Retriever` protocol](../reference/api.md#realm_retrieve.models.retriever):

```python
class MyRetriever:
    def retrieve(self, query: str, k: int = 5, return_scores: bool = False) -> list[dict]: ...
    def get_corpus_size(self) -> int: ...
```
