"""Base classes for prediction results and baseline evaluation runners."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalEvent:
    """A single retrieval event during reasoning."""

    step_index: int
    position_fraction: float
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "position_fraction": self.position_fraction,
            "latency_ms": self.latency_ms,
        }


@dataclass
class PredictionResult:
    """Complete prediction record for one question."""

    question_id: str
    dataset: str
    method: str
    seed: int
    model: str
    gold_answer: str
    predicted_answer: str
    em: float
    f1: float
    num_retrieval_calls: int
    retrieval_events: List[RetrievalEvent] = field(default_factory=list)
    reasoning_tokens: int = 0
    e2e_latency_s: float = 0.0
    num_hops: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "question_id": self.question_id,
            "dataset": self.dataset,
            "method": self.method,
            "seed": self.seed,
            "model": self.model,
            "gold_answer": self.gold_answer,
            "predicted_answer": self.predicted_answer,
            "em": self.em,
            "f1": round(self.f1, 4),
            "num_retrieval_calls": self.num_retrieval_calls,
            "retrieval_events": [e.to_dict() for e in self.retrieval_events],
            "reasoning_tokens": self.reasoning_tokens,
            "e2e_latency_s": self.e2e_latency_s,
        }
        if self.num_hops is not None:
            d["num_hops"] = self.num_hops
        return d


_ANSWER_PATTERNS = [
    re.compile(r"[Tt]he answer is[:\s]+(.+?)[\.\n]"),
    re.compile(r"[Ff]inal [Aa]nswer[:\s]+(.+?)[\.\n]"),
    re.compile(r"[Aa]nswer[:\s]+(.+?)[\.\n]"),
]


def extract_answer(reasoning_chain: str) -> str:
    for pat in _ANSWER_PATTERNS:
        matches = pat.findall(reasoning_chain)
        if matches:
            return matches[-1].strip()
    sentences = reasoning_chain.split(".")
    return sentences[-2].strip() if len(sentences) > 1 else ""


class BaseRunner:
    """Base class for baseline evaluation methods.

    Subclasses must implement ``run_question`` and should compute RSUS
    independently from their own reasoning chain by calling
    ``_compute_step_rsus``.
    """

    def __init__(self, reasoning_model, retriever, config, rsus_calculator=None):
        self.reasoning_model = reasoning_model
        self.retriever = retriever
        self.config = config
        self.rsus_calculator = rsus_calculator

    def _compute_step_rsus(self, steps, compute_consistency: bool = False):
        """Compute per-step RSUS (used internally for policy decisions)."""
        if self.rsus_calculator is None:
            return []
        results = []
        for i, step in enumerate(steps):
            text = step.text if hasattr(step, "text") else str(step)
            context = "\n".join(
                s.text if hasattr(s, "text") else str(s) for s in steps[:i]
            )
            score, _ = self.rsus_calculator.compute_rsus(
                reasoning_step=text,
                context=context,
                compute_consistency=compute_consistency,
            )
            results.append({"step_index": i, "rsus_composite": score})
        return results

    def run_question(
        self, question, question_id, dataset, gold_answer, seed, **kwargs
    ) -> PredictionResult:
        raise NotImplementedError


_BASELINE_REGISTRY: Dict[str, type] = {}


def register_baseline(name: str):
    def decorator(cls):
        _BASELINE_REGISTRY[name] = cls
        return cls
    return decorator


def get_baseline_class(name: str) -> type:
    if name not in _BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline {name!r}. "
            f"Registered: {sorted(_BASELINE_REGISTRY)}"
        )
    return _BASELINE_REGISTRY[name]
