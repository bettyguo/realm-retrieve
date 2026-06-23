"""Self-contained, dependency-free demonstration of the ReaLM-Retrieve loop.

This module mirrors the four-stage pipeline (segment -> score -> decide ->
retrieve) using only the Python standard library, so the end-to-end control
flow can be exercised on CPU without torch, spaCy, vLLM, or ColBERT. The full
implementations live in :mod:`realm_retrieve.models`; the classes here are the
miniature counterparts that drive the ``realm-quickstart`` console script and
the unit tests.

    from realm_retrieve.toy import ToyPipeline, ToyRetriever, ToyReasoningModel
    pipe = ToyPipeline(ToyRetriever(demo_corpus()), ToyReasoningModel(seed=0))
    result = pipe.answer("What is the capital of Sweden?", gold="Stockholm")
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Text utilities (token-level EM / F1, matching evaluation.metrics semantics)
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, gold: str) -> bool:
    return _normalize(prediction) == _normalize(gold)


_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
        "is", "are", "was", "were", "be", "by", "with", "as", "that", "this",
        "what", "which", "who", "whom", "did", "do", "does", "how", "when",
        "where", "why", "it", "its", "into", "from", "not",
    }
)


def _terms(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Corpus + retriever
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToyDocument:
    """An immutable passage in the toy corpus."""

    passage_id: str
    text: str
    answer: str = ""


class ToyRetriever:
    """A pure-Python BM25 retriever over a small in-memory corpus."""

    def __init__(self, corpus: List[ToyDocument], k1: float = 1.5, b: float = 0.75):
        self.corpus: List[ToyDocument] = list(corpus)
        self.k1 = k1
        self.b = b
        self._by_id: Dict[str, ToyDocument] = {d.passage_id: d for d in self.corpus}

        self._doc_terms: List[List[str]] = [_terms(d.text) for d in self.corpus]
        self._doc_len: List[int] = [len(t) for t in self._doc_terms]
        self._avg_len: float = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0

        self._df: Counter = Counter()
        for terms in self._doc_terms:
            for term in set(terms):
                self._df[term] += 1

    def get_corpus_size(self) -> int:
        return len(self.corpus)

    def document(self, passage_id: str) -> Optional[ToyDocument]:
        return self._by_id.get(passage_id)

    def _idf(self, term: str) -> float:
        n = self._df.get(term, 0)
        if n == 0:
            return 0.0
        total = len(self.corpus)
        # Always-positive Lucene-style idf so common terms cannot flip ranking.
        return math.log(1.0 + (total - n + 0.5) / (n + 0.5))

    def _score(self, query_terms: List[str], doc_index: int) -> float:
        terms = self._doc_terms[doc_index]
        if not terms:
            return 0.0
        tf = Counter(terms)
        length = self._doc_len[doc_index]
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            idf = self._idf(term)
            denom = freq + self.k1 * (1 - self.b + self.b * length / (self._avg_len or 1.0))
            score += idf * (freq * (self.k1 + 1)) / denom
        return score

    def retrieve(
        self,
        query: str,
        k: int = 5,
        return_scores: bool = False,
    ) -> List[Dict]:
        if not self.corpus:
            return []
        query_terms = _terms(query)
        scored: List[Tuple[int, float]] = []
        for idx in range(len(self.corpus)):
            score = self._score(query_terms, idx)
            if score > 0.0:
                scored.append((idx, score))
        # Sort by score (desc), break ties by corpus order for determinism.
        scored.sort(key=lambda pair: (-pair[1], pair[0]))

        results: List[Dict] = []
        for rank, (idx, score) in enumerate(scored[:k]):
            doc = self.corpus[idx]
            entry: Dict = {
                "passage_id": doc.passage_id,
                "text": doc.text,
                "rank": rank,
            }
            if return_scores:
                entry["score"] = score
            results.append(entry)
        return results


# ---------------------------------------------------------------------------
# Reasoning model + pipeline
# ---------------------------------------------------------------------------


@dataclass
class ToyResult:
    """Outcome of running one question through the toy pipeline."""

    question: str
    answer: str
    gold: str
    f1: float
    correct: bool
    retrievals: int
    rsus_per_step: List[float] = field(default_factory=list)
    retrieved_doc_ids: List[str] = field(default_factory=list)


class ToyReasoningModel:
    """Deterministic stand-in for a large reasoning model.

    Emits a fixed two-step reasoning sketch: a committed planning step (low
    uncertainty) followed by a hedged verification step (high uncertainty) that
    invites retrieval. The ``seed`` is recorded for interface parity with the
    full wrapper; the toy outputs are fully deterministic.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def reason(self, question: str) -> List[str]:
        return [
            f"First, I restate the problem and outline the approach for: {question}",
            "I am not sure about the key fact yet; let me verify it to confirm the answer.",
        ]

    def read(self, documents: List[Dict], retriever: ToyRetriever) -> str:
        for doc in documents:
            source = retriever.document(doc["passage_id"])
            if source is not None and source.answer:
                return source.answer
        return ""


class ToyPipeline:
    """Miniature segment -> RSUS -> policy -> retrieve loop."""

    _HEDGE = (
        "not sure", "unsure", "uncertain", "unclear", "maybe", "perhaps",
        "possibly", "might", "could", "verify", "let me check", "confirm",
    )
    _COMMIT = (
        "clearly", "obviously", "definitely", "certainly", "established fact",
        "no ambiguity", "without doubt", "i know",
    )

    def __init__(
        self,
        retriever: ToyRetriever,
        reasoner: ToyReasoningModel,
        rsus_threshold: float = 0.5,
        k: int = 3,
    ):
        self.retriever = retriever
        self.reasoner = reasoner
        self.rsus_threshold = rsus_threshold
        self.k = k

    def rsus(self, step_text: str) -> float:
        """Heuristic step-level uncertainty in ``[0, 1]``.

        Starts neutral and shifts up for hedging markers, down for committed
        ones, so committed prose scores below the threshold and hedged prose
        scores above it.
        """
        lowered = step_text.lower()
        hedge = sum(marker in lowered for marker in self._HEDGE)
        commit = sum(marker in lowered for marker in self._COMMIT)
        score = 0.5 + 0.2 * hedge - 0.2 * commit
        return max(0.0, min(1.0, score))

    def policy(self, rsus_score: float) -> bool:
        """Retrieve only when uncertainty exceeds the threshold."""
        return rsus_score > self.rsus_threshold

    def answer(self, question: str, gold: str = "") -> ToyResult:
        steps = self.reasoner.reason(question)
        rsus_per_step: List[float] = []
        retrieved_doc_ids: List[str] = []
        retrievals = 0
        prediction = ""

        for step in steps:
            score = self.rsus(step)
            rsus_per_step.append(score)
            if self.policy(score):
                retrievals += 1
                hits = self.retriever.retrieve(question, k=self.k)
                retrieved_doc_ids.extend(hit["passage_id"] for hit in hits)
                if not prediction:
                    prediction = self.reasoner.read(hits, self.retriever)

        return ToyResult(
            question=question,
            answer=prediction,
            gold=gold,
            f1=_f1(prediction, gold),
            correct=_exact_match(prediction, gold),
            retrievals=retrievals,
            rsus_per_step=rsus_per_step,
            retrieved_doc_ids=retrieved_doc_ids,
        )


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------


def demo_corpus() -> List[ToyDocument]:
    return [
        ToyDocument("d01", "The Berlin Wall fell in 1989, ending the division of Germany.", "1989"),
        ToyDocument("d02", "Beijing hosted the Summer Olympics in 2008 and the Winter Olympics in 2022.", "Beijing"),
        ToyDocument("d03", "Stockholm is the capital of Sweden, built across fourteen islands.", "Stockholm"),
        ToyDocument("d04", "Paris is the capital of France and lies on the river Seine.", "Paris"),
        ToyDocument("d05", "Mount Everest is the highest mountain above sea level on Earth.", "Mount Everest"),
        ToyDocument("d06", "Water is a molecule of hydrogen and oxygen with the formula H2O.", "H2O"),
        ToyDocument("d07", "Albert Einstein developed the theory of relativity in modern physics.", "Albert Einstein"),
        ToyDocument("d08", "The Pacific Ocean is the largest and deepest ocean on Earth.", "Pacific Ocean"),
        ToyDocument("d09", "Canberra is the capital city of Australia, located between Sydney and Melbourne.", "Canberra"),
        ToyDocument("d10", "The Amazon river in South America discharges more water than any other river.", "Amazon"),
        ToyDocument("d11", "Python is a high-level programming language created by Guido van Rossum.", "Python"),
        ToyDocument("d12", "ColBERT is a neural retriever that scores documents with late interaction over BERT embeddings.", "ColBERT"),
    ]


def demo_questions() -> List[Tuple[str, str]]:
    return [
        ("In what year did the Berlin Wall fall?", "1989"),
        ("What is the capital of Sweden?", "Stockholm"),
        ("What is the capital city of Australia?", "Canberra"),
        ("Who developed the theory of relativity?", "Albert Einstein"),
        ("Which ocean is the largest on Earth?", "Pacific Ocean"),
    ]
