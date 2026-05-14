"""
Lightweight, CPU-only toy implementations of the ReaLM-Retrieve components.

The real models in :mod:`realm_retrieve.models` require ColBERTv2, vLLM, and a
multi-GPU box. This module is a *teaching* implementation that captures the
shape of the pipeline (segment → RSUS → policy → retrieve → answer) using only
the Python standard library + NumPy, so newcomers can run the full loop in
seconds.

The interfaces deliberately match the real classes' contracts (`.retrieve`,
`.generate`, `.segment`, …) so that swapping toy → real is a one-line change.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

__all__ = [
    "ToyDocument",
    "ToyRetriever",
    "ToyReasoningModel",
    "ToyPipeline",
    "ToyResult",
    "demo_corpus",
    "demo_questions",
]


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToyDocument:
    """A single corpus document used by the toy pipeline."""

    doc_id: str
    text: str


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class ToyRetriever:
    """A pure-Python BM25 retriever.

    Mirrors the public surface of
    :class:`realm_retrieve.models.retriever.ColBERTRetriever` so the pipeline
    code can stay identical between the toy and production paths.
    """

    def __init__(
        self,
        corpus: Iterable[ToyDocument],
        k: int = 3,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.corpus: list[ToyDocument] = list(corpus)
        self.k = k
        self.k1 = k1
        self.b = b

        self._tokens: list[list[str]] = [_tokenise(d.text) for d in self.corpus]
        self._lengths: list[int] = [len(toks) for toks in self._tokens]
        self._avgdl: float = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

        self._df: Counter[str] = Counter()
        for toks in self._tokens:
            for term in set(toks):
                self._df[term] += 1
        self._N = len(self.corpus)

    # -- public API ------------------------------------------------------- #

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        return_scores: bool = False,
    ) -> list[dict]:
        if k is None:
            k = self.k

        query_terms = _tokenise(query)
        scored: list[tuple[float, int]] = []
        for idx, toks in enumerate(self._tokens):
            score = self._bm25(query_terms, toks, self._lengths[idx])
            if score > 0:
                scored.append((score, idx))

        scored.sort(reverse=True)
        top = scored[:k]
        results: list[dict] = []
        for rank, (score, idx) in enumerate(top):
            doc = self.corpus[idx]
            entry = {
                "passage_id": doc.doc_id,
                "text": doc.text,
                "rank": rank,
            }
            if return_scores:
                entry["score"] = score
            results.append(entry)
        return results

    def get_corpus_size(self) -> int:
        return len(self.corpus)

    # -- internals -------------------------------------------------------- #

    def _bm25(self, q_terms: list[str], d_terms: list[str], dl: int) -> float:
        if dl == 0 or not q_terms:
            return 0.0
        tf = Counter(d_terms)
        score = 0.0
        for term in q_terms:
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)
            f = tf[term]
            num = f * (self.k1 + 1)
            den = f + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1e-9))
            score += idf * num / den
        return score


# --------------------------------------------------------------------------- #
# Reasoning model
# --------------------------------------------------------------------------- #


class ToyReasoningModel:
    """A deterministic toy 'LRM'.

    It produces a short reasoning chain made of two steps:

    1. A *hypothesis* step composed from the question's key nouns.
    2. A *verification* step that copies the most relevant retrieved passage
       (if any) and extracts the answer span via string overlap.

    Crucially it is fully deterministic given a seed, so the quickstart is
    reproducible across machines.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def generate_reasoning(self, question: str) -> list[str]:
        """Return a two-step reasoning chain for *question*.

        Step 1 is deliberately hedged (high RSUS → policy retrieves).
        Step 2 is committed (low RSUS → policy skips). This contrast is what
        makes the quickstart visually demonstrate the policy at work.
        """
        keywords = " ".join(_tokenise(question)[:6])
        step1 = (
            f"Let me think about {keywords}. I am not sure about the exact "
            "fact and I'm not certain which detail is right. Perhaps I should "
            "verify this — I'm not fully confident without checking the source."
        )
        step2 = (
            "Therefore, having reviewed the retrieved context, I can now "
            "confidently state the final answer below."
        )
        return [step1, step2]

    def answer(self, question: str, retrieved: list[dict]) -> str:
        """Extract a best-effort answer from the question + retrieved passages.

        Picks the passage with the highest term overlap with the question,
        then applies a small set of question-type heuristics to extract a span:

        - *year / when* questions → first 4-digit number.
        - *who* questions → first proper-noun bigram.
        - *where / city / capital* questions → first proper noun not also in
          the question.
        - otherwise → last proper-noun span.
        """
        if not retrieved:
            return "unknown"

        q_terms = set(_tokenise(question))
        q_lower = question.lower()

        # Best passage = highest token overlap with the question.
        best_text = ""
        best_overlap = -1
        for doc in retrieved:
            terms = set(_tokenise(doc["text"]))
            overlap = len(q_terms & terms)
            if overlap > best_overlap:
                best_overlap = overlap
                best_text = doc["text"]

        # Year / date questions.
        if any(k in q_lower for k in (" year", "when ", "what year", "which year")):
            years = re.findall(r"\b(1[5-9]\d{2}|20\d{2})\b", best_text)
            if years:
                return years[0]

        proper_nouns = re.findall(
            r"\b([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)*)\b", best_text
        )
        # Drop entities that overlap heavily with the question (likely the
        # subject, not the answer).
        novel = [
            p for p in proper_nouns
            if not set(_tokenise(p)).issubset(q_terms)
        ]
        candidates = novel or proper_nouns

        # Who → prefer multi-word names.
        if q_lower.startswith("who "):
            multi = [c for c in candidates if " " in c]
            if multi:
                return multi[0]

        # Where / which city / capital → first novel proper noun.
        if any(k in q_lower for k in ("where", "which city", "what city", "capital")):
            return candidates[0] if candidates else "unknown"

        return candidates[0] if candidates else best_text.split(".")[0]


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


@dataclass
class ToyResult:
    question: str
    answer: str
    gold: str
    correct: bool
    rsus_per_step: list[float]
    retrievals: int
    retrieved_doc_ids: list[str] = field(default_factory=list)
    f1: float = 0.0

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        check = "✓" if self.correct else "✗"
        return (
            f"Q: {self.question}\n"
            f"  RSUS={['%.2f' % s for s in self.rsus_per_step]}  "
            f"retrievals={self.retrievals}\n"
            f"  → {self.answer}   (gold: {self.gold}) {check}"
        )


class ToyPipeline:
    """Tiny end-to-end pipeline matching the real ReaLM-Retrieve loop.

    The retrieval policy is a transparent threshold over the RSUS score
    (instead of a learned REINFORCE network), so the demo is fully
    deterministic and free of training cost.
    """

    def __init__(
        self,
        retriever: ToyRetriever,
        reasoner: ToyReasoningModel,
        rsus_threshold: float = 0.5,
        alpha: float = 0.4,
        beta: float = 0.35,
        gamma: float = 0.25,
    ) -> None:
        self.retriever = retriever
        self.reasoner = reasoner
        self.threshold = rsus_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    # -- RSUS ------------------------------------------------------------- #

    def rsus(self, step_text: str) -> float:
        """Compute a deterministic toy RSUS in [0, 1]."""
        text = step_text.lower()

        # U_verb: presence of hedging markers
        hedges = (
            "not sure", "i think", "might", "perhaps", "maybe", "uncertain",
            "verify", "let me check", "i'm not", "i am not", "should i",
        )
        u_verb = min(1.0, sum(1 for h in hedges if h in text) / 3.0)

        # U_ent: density of capitalised entities (proxy for entity entropy)
        entities = re.findall(r"\b[A-Z][a-zA-Z]+\b", step_text)
        u_ent = min(1.0, len(entities) / 8.0)

        # U_cons: very short steps are unstable, long ones are usually committed
        tok_n = max(1, len(step_text.split()))
        u_cons = min(1.0, 1.0 - tok_n / 60.0) if tok_n < 60 else 0.1

        return self.alpha * u_verb + self.beta * u_ent + self.gamma * u_cons

    def policy(self, rsus_score: float) -> bool:
        """Toy retrieval-intervention policy = threshold on RSUS."""
        return rsus_score >= self.threshold

    # -- main loop -------------------------------------------------------- #

    def answer(self, question: str, gold: str = "") -> ToyResult:
        steps = self.reasoner.generate_reasoning(question)
        scores: list[float] = []
        retrieved: list[dict] = []
        n_retrievals = 0

        for step in steps:
            score = self.rsus(step)
            scores.append(score)
            if self.policy(score):
                n_retrievals += 1
                retrieved.extend(self.retriever.retrieve(f"{question} {step}", k=3))

        answer = self.reasoner.answer(question, retrieved)
        gold_norm = gold.strip().lower()
        ans_norm = answer.strip().lower()
        correct = bool(gold_norm) and ans_norm == gold_norm
        f1 = _token_f1(answer, gold)

        return ToyResult(
            question=question,
            answer=answer,
            gold=gold,
            correct=correct,
            rsus_per_step=scores,
            retrievals=n_retrievals,
            retrieved_doc_ids=[d["passage_id"] for d in retrieved],
            f1=f1,
        )


def _token_f1(prediction: str, ground_truth: str) -> float:
    pred = _tokenise(prediction)
    gold = _tokenise(ground_truth)
    if not pred or not gold:
        return float(pred == gold)
    common = Counter(pred) & Counter(gold)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p = same / len(pred)
    r = same / len(gold)
    return 2 * p * r / (p + r)


# --------------------------------------------------------------------------- #
# Demo data
# --------------------------------------------------------------------------- #


def demo_corpus() -> list[ToyDocument]:
    """A 12-document toy corpus covering the demo questions."""
    return [
        ToyDocument("d01", "The Berlin Wall fell on 9 November 1989, ending decades of separation in Germany."),
        ToyDocument("d02", "Beijing hosted the 2008 Summer Olympic Games, marking China's first time as host."),
        ToyDocument("d03", "Stockholm is the capital and largest city of Sweden, located on the Baltic Sea."),
        ToyDocument("d04", "Sweden won the 1958 FIFA World Cup runner-up title, played at home."),
        ToyDocument("d05", "Marie Curie was awarded the Nobel Prize in Physics in 1903 and in Chemistry in 1911."),
        ToyDocument("d06", "The Eiffel Tower was completed in 1889 and is located in Paris, France."),
        ToyDocument("d07", "Mount Everest, on the border of Nepal and Tibet, rises to 8,848 metres."),
        ToyDocument("d08", "Alan Turing introduced the concept of the universal computing machine in 1936."),
        ToyDocument("d09", "The capital of Australia is Canberra, not Sydney as commonly assumed."),
        ToyDocument("d10", "Tokyo became the capital of Japan during the Meiji Restoration in 1868."),
        ToyDocument("d11", "The Amazon River, primarily in Brazil, is the largest river by discharge."),
        ToyDocument("d12", "ColBERT is a late-interaction neural retriever proposed at SIGIR 2020."),
    ]


def demo_questions() -> list[tuple[str, str]]:
    """Return (question, gold-answer) pairs that exercise both branches of the
    policy (retrieve vs. skip)."""
    return [
        ("In what year did the Berlin Wall fall?", "1989"),
        ("Which city hosted the 2008 Summer Olympics?", "Beijing"),
        ("What is the capital of Sweden?", "Stockholm"),
        ("Who introduced the universal computing machine?", "Alan Turing"),
        ("What is the capital of Australia?", "Canberra"),
    ]
