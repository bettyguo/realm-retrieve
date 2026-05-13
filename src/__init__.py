"""
ReaLM-Retrieve: Adaptive Retrieval for Large Reasoning Models

When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models
SIGIR 2026

This package implements a reasoning-aware retrieval framework that addresses the temporal
mismatch between large reasoning models (LRMs) and traditional RAG systems through:
1. Step-level uncertainty detection (RSUS)
2. Learned retrieval intervention policy (REINFORCE)
3. Efficiency-optimized integration (3.2× per-call reduction)
"""

__version__ = "1.0.0"
__author__ = "Anonymous Author(s)"
__license__ = "Apache-2.0"

# Core exports
from realm_retrieve.models.segmentation import ReasoningStepSegmenter
from realm_retrieve.models.rsus import RSUSCalculator
from realm_retrieve.models.policy import RetrievalInterventionPolicy
from realm_retrieve.models.retriever import ColBERTRetriever
from realm_retrieve.models.reasoning_model import ReasoningModelWrapper

__all__ = [
    "ReasoningStepSegmenter",
    "RSUSCalculator",
    "RetrievalInterventionPolicy",
    "ColBERTRetriever",
    "ReasoningModelWrapper",
]
