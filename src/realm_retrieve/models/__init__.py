"""Models module — lazy-loaded so CPU-only consumers can still ``import realm_retrieve``."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY: dict[str, tuple[str, str]] = {
    "ReasoningStepSegmenter":      ("realm_retrieve.models.segmentation",     "ReasoningStepSegmenter"),
    "ReasoningStep":               ("realm_retrieve.models.segmentation",     "ReasoningStep"),
    "RSUSCalculator":              ("realm_retrieve.models.rsus",             "RSUSCalculator"),
    "RSUSComponents":              ("realm_retrieve.models.rsus",             "RSUSComponents"),
    "RetrievalInterventionPolicy": ("realm_retrieve.models.policy",           "RetrievalInterventionPolicy"),
    "REINFORCETrainer":            ("realm_retrieve.models.policy",           "REINFORCETrainer"),
    "PolicyState":                 ("realm_retrieve.models.policy",           "PolicyState"),
    "PolicyAction":                ("realm_retrieve.models.policy",           "PolicyAction"),
    "ColBERTRetriever":            ("realm_retrieve.models.retriever",        "ColBERTRetriever"),
    "ReasoningModelWrapper":       ("realm_retrieve.models.reasoning_model",  "ReasoningModelWrapper"),
    "VLLMReasoningModel":          ("realm_retrieve.models.reasoning_model",  "VLLMReasoningModel"),
    "OpenAIReasoningModel":        ("realm_retrieve.models.reasoning_model",  "OpenAIReasoningModel"),
    "create_reasoning_model":      ("realm_retrieve.models.reasoning_model",  "create_reasoning_model"),
    "QueryGen":                    ("realm_retrieve.models.query_gen",        "QueryGen"),
    "ProxyMLP":                    ("realm_retrieve.models.proxy_mlp",        "ProxyMLP"),
    "ImplicitCompression":         ("realm_retrieve.models.compression",      "ImplicitCompression"),
    "CompressedResult":            ("realm_retrieve.models.compression",      "CompressedResult"),
    "SpeculativeCache":            ("realm_retrieve.models.speculative_cache", "SpeculativeCache"),
    "KVCacheManager":              ("realm_retrieve.models.kv_cache",         "KVCacheManager"),
    "RetrievalInjector":           ("realm_retrieve.models.injection",        "RetrievalInjector"),
    "InjectionResult":             ("realm_retrieve.models.injection",        "InjectionResult"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'realm_retrieve.models' has no attribute {name!r}")
    submodule = importlib.import_module(target[0])
    value = getattr(submodule, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover
    return sorted({*globals(), *_LAZY})


if TYPE_CHECKING:  # pragma: no cover
    from realm_retrieve.models.compression import CompressedResult, ImplicitCompression
    from realm_retrieve.models.injection import InjectionResult, RetrievalInjector
    from realm_retrieve.models.kv_cache import KVCacheManager
    from realm_retrieve.models.policy import (
        PolicyAction,
        PolicyState,
        REINFORCETrainer,
        RetrievalInterventionPolicy,
    )
    from realm_retrieve.models.proxy_mlp import ProxyMLP
    from realm_retrieve.models.query_gen import QueryGen
    from realm_retrieve.models.reasoning_model import (
        OpenAIReasoningModel,
        ReasoningModelWrapper,
        VLLMReasoningModel,
        create_reasoning_model,
    )
    from realm_retrieve.models.retriever import ColBERTRetriever
    from realm_retrieve.models.rsus import RSUSCalculator, RSUSComponents
    from realm_retrieve.models.segmentation import ReasoningStep, ReasoningStepSegmenter
    from realm_retrieve.models.speculative_cache import SpeculativeCache


__all__ = list(_LAZY.keys())
