"""Tests for the pure-Python BM25 toy retriever."""

from __future__ import annotations

import pytest

from realm_retrieve.toy import ToyDocument, ToyRetriever


class TestToyRetriever:
    def test_corpus_size(self, retriever, corpus):
        assert retriever.get_corpus_size() == len(corpus)

    def test_retrieve_returns_up_to_k(self, retriever):
        results = retriever.retrieve("Berlin", k=3)
        assert 1 <= len(results) <= 3

    def test_retrieves_relevant_doc_first(self, retriever):
        results = retriever.retrieve("Berlin Wall fall year", k=1)
        assert results[0]["passage_id"] == "d01"

    def test_returns_scores_when_asked(self, retriever):
        results = retriever.retrieve("Olympics Beijing", k=2, return_scores=True)
        assert "score" in results[0]
        assert results[0]["score"] >= results[-1]["score"]  # monotone non-increasing

    def test_no_results_for_completely_unrelated_query(self, retriever):
        # gibberish that doesn't match any term in the toy corpus
        assert retriever.retrieve("xyzzy plugh fnord", k=5) == []

    def test_handles_empty_corpus(self):
        r = ToyRetriever([])
        assert r.retrieve("anything") == []
        assert r.get_corpus_size() == 0

    @pytest.mark.parametrize(
        "query, expected_doc",
        [
            ("capital of Sweden", "d03"),
            ("Australia capital city", "d09"),
            ("ColBERT retriever", "d12"),
        ],
    )
    def test_known_queries(self, retriever, query, expected_doc):
        results = retriever.retrieve(query, k=1)
        assert results[0]["passage_id"] == expected_doc


class TestToyDocumentEquality:
    def test_frozen_dataclass(self):
        d = ToyDocument("x", "hello")
        with pytest.raises(Exception):
            d.text = "world"  # type: ignore[misc]
