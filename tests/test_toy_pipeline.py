"""End-to-end tests for the toy pipeline that drives the quickstart demo."""

from __future__ import annotations

import pytest

from realm_retrieve.toy import (
    ToyPipeline,
    ToyReasoningModel,
    ToyResult,
    ToyRetriever,
    demo_corpus,
    demo_questions,
)


class TestRSUS:
    def test_low_for_committed_text(self, pipeline):
        text = (
            "Clearly the answer is Beijing. This is an established fact "
            "and there is no ambiguity here at all in any way."
        )
        assert pipeline.rsus(text) < 0.5

    def test_high_for_hedged_text(self, pipeline):
        text = "I am not sure. Maybe I should verify. Perhaps the answer is X."
        assert pipeline.rsus(text) > 0.5


class TestPolicy:
    def test_below_threshold_skips(self, pipeline):
        assert pipeline.policy(0.1) is False

    def test_above_threshold_retrieves(self, pipeline):
        assert pipeline.policy(0.9) is True


class TestPipelineEndToEnd:
    def test_returns_toy_result(self, pipeline):
        out = pipeline.answer("In what year did the Berlin Wall fall?", gold="1989")
        assert isinstance(out, ToyResult)
        assert out.question.startswith("In what year")

    def test_retrieval_count_matches_recorded_docs(self, pipeline):
        out = pipeline.answer("What is the capital of Sweden?", gold="Stockholm")
        # Each retrieval call fetches up to 3 docs.
        assert len(out.retrieved_doc_ids) <= 3 * out.retrievals + 1

    def test_quickstart_baseline_quality(self):
        """Sanity floor: the toy pipeline should answer at least one demo
        question correctly. This guards against accidental regressions in
        the demo content."""
        retriever = ToyRetriever(demo_corpus())
        pipe = ToyPipeline(retriever, ToyReasoningModel(seed=0))
        correct = 0
        for q, gold in demo_questions():
            r = pipe.answer(q, gold=gold)
            if r.f1 > 0.0:
                correct += 1
        assert correct >= 1


class TestPipelineDeterminism:
    def test_same_seed_gives_same_answer(self):
        retriever = ToyRetriever(demo_corpus())
        p1 = ToyPipeline(retriever, ToyReasoningModel(seed=7))
        p2 = ToyPipeline(retriever, ToyReasoningModel(seed=7))
        q, gold = demo_questions()[0]
        assert p1.answer(q, gold=gold).answer == p2.answer(q, gold=gold).answer
