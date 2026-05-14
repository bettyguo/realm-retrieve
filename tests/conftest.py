"""Shared pytest fixtures for the ReaLM-Retrieve test suite."""

from __future__ import annotations

import pytest

from realm_retrieve.toy import (
    ToyPipeline,
    ToyReasoningModel,
    ToyRetriever,
    demo_corpus,
    demo_questions,
)


@pytest.fixture(scope="session")
def corpus():
    return demo_corpus()


@pytest.fixture(scope="session")
def questions():
    return demo_questions()


@pytest.fixture()
def retriever(corpus):
    return ToyRetriever(corpus)


@pytest.fixture()
def pipeline(retriever):
    return ToyPipeline(retriever, ToyReasoningModel(seed=0), rsus_threshold=0.5)
