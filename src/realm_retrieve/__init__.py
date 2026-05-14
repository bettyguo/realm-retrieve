"""
ReaLM-Retrieve: Adaptive Retrieval for Large Reasoning Models.

When to Retrieve During Reasoning — SIGIR 2026.

Heavy dependencies (vLLM, ColBERT, faiss-gpu) are *lazy-imported* so that
``import realm_retrieve`` works on CPU-only environments and the
:func:`realm_retrieve.toy` quickstart runs without any of them installed.

CPU-only path::

    from realm_retrieve.toy import ToyPipeline, ToyRetriever, ToyReasoningModel

Full GPU path::

    from realm_retrieve import (
        ReasoningStepSegmenter, RSUSCalculator,
        RetrievalInterventionPolicy, ColBERTRetriever,
    )
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"
__author__ = "Dongxin Guo, Jikun Wu, Siu Ming Yiu"
__license__ = "Apache-2.0"


# ---------------------------------------------------------------------------
# Lazy attribute access — keeps `import realm_retrieve` fast and avoids
# pulling vLLM / ColBERT / spaCy unless the caller actually needs them.
# ---------------------------------------------------------------------------

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ReasoningStepSegmenter":      ("realm_retrieve.models.segmentation", "ReasoningStepSegmenter"),
    "ReasoningStep":               ("realm_retrieve.models.segmentation", "ReasoningStep"),
    "RSUSCalculator":              ("realm_retrieve.models.rsus",         "RSUSCalculator"),
    "RSUSComponents":              ("realm_retrieve.models.rsus",         "RSUSComponents"),
    "RetrievalInterventionPolicy": ("realm_retrieve.models.policy",       "RetrievalInterventionPolicy"),
    "REINFORCETrainer":            ("realm_retrieve.models.policy",       "REINFORCETrainer"),
    "PolicyState":                 ("realm_retrieve.models.policy",       "PolicyState"),
    "PolicyAction":                ("realm_retrieve.models.policy",       "PolicyAction"),
    "ColBERTRetriever":            ("realm_retrieve.models.retriever",    "ColBERTRetriever"),
    "ReasoningModelWrapper":       ("realm_retrieve.models.reasoning_model", "ReasoningModelWrapper"),
    "VLLMReasoningModel":          ("realm_retrieve.models.reasoning_model", "VLLMReasoningModel"),
    "OpenAIReasoningModel":        ("realm_retrieve.models.reasoning_model", "OpenAIReasoningModel"),
    "create_reasoning_model":      ("realm_retrieve.models.reasoning_model", "create_reasoning_model"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module 'realm_retrieve' has no attribute {name!r}")
    submodule_name, attr_name = target
    submodule = importlib.import_module(submodule_name)
    value = getattr(submodule, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover
    return sorted({*globals(), *_LAZY_ATTRS})


if TYPE_CHECKING:  # pragma: no cover
    from realm_retrieve.models.policy import (
        PolicyAction,
        PolicyState,
        REINFORCETrainer,
        RetrievalInterventionPolicy,
    )
    from realm_retrieve.models.reasoning_model import (
        OpenAIReasoningModel,
        ReasoningModelWrapper,
        VLLMReasoningModel,
        create_reasoning_model,
    )
    from realm_retrieve.models.retriever import ColBERTRetriever
    from realm_retrieve.models.rsus import RSUSCalculator, RSUSComponents
    from realm_retrieve.models.segmentation import (
        ReasoningStep,
        ReasoningStepSegmenter,
    )


__all__ = [
    "__version__",
    "ColBERTRetriever",
    "OpenAIReasoningModel",
    "PolicyAction",
    "PolicyState",
    "REINFORCETrainer",
    "ReasoningModelWrapper",
    "ReasoningStep",
    "ReasoningStepSegmenter",
    "RetrievalInterventionPolicy",
    "RSUSCalculator",
    "RSUSComponents",
    "VLLMReasoningModel",
    "create_reasoning_model",
]
