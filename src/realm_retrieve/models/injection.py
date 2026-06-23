# SPDX-License-Identifier: Apache-2.0
"""
Full Injection Protocol

Replaces the rudimentary evidence injection in ``evaluate.py`` with
the complete protocol described in the paper:

1. Pause generation at a step boundary (from the segmentation classifier)
2. Optionally compress evidence via ImplicitCompression
3. Wrap evidence in ``<retrieved>`` / ``</retrieved>`` delimiters
4. For open-weight models: use KVCacheManager to preserve prefix cache
5. For API models: standard concatenation

Injection occurs only at step boundaries, never mid-step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from realm_retrieve.models.kv_cache import KVCacheManager
from realm_retrieve.models.compression import ImplicitCompression
from realm_retrieve.models.speculative_cache import SpeculativeCache

import numpy as np


EVIDENCE_PREFIX = "\n<retrieved>\n"
EVIDENCE_SUFFIX = "\n</retrieved>\n"
CONTINUATION_PROMPT = "Continue your reasoning, using this evidence if relevant."


@dataclass
class InjectionResult:
    """Output of the injection protocol."""
    augmented_input: str
    prefix_length: int
    compression_ratio: float


class RetrievalInjector:
    """
    Inject retrieved evidence into a reasoning chain at step boundaries.

    Supports two code-paths:
    * **open-weight** (``model_type="vllm"``): delegates to ``KVCacheManager``
      for prefix-cache reuse.
    * **api** (``model_type="api"`` or any other value): plain string
      concatenation.
    """

    def __init__(
        self,
        kv_cache_manager: Optional[KVCacheManager] = None,
        compression: Optional[ImplicitCompression] = None,
        speculative_cache: Optional[SpeculativeCache] = None,
    ):
        self.kv_cache_manager = kv_cache_manager
        self.compression = compression
        self.speculative_cache = speculative_cache

    def inject(
        self,
        prefix: str,
        evidence: List[str],
        model_type: str,
        tokenizer: Any = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> InjectionResult:
        """
        Build the augmented input after injecting evidence.

        Args:
            prefix: reasoning chain up to step boundary ``r_{1:i}``
            evidence: list of retrieved passage strings
            model_type: ``"vllm"`` for open-weight, anything else for API
            tokenizer: required when *model_type* is ``"vllm"``
            query_embedding: required when compression is enabled

        Returns:
            InjectionResult
        """
        compression_ratio = 0.0

        if self.compression is not None and query_embedding is not None:
            result = self.compression.compress(evidence, query_embedding)
            evidence = result.compressed_passages
            compression_ratio = result.compression_ratio

        evidence_block = "\n\n".join(evidence)
        augmented_input = (
            prefix
            + EVIDENCE_PREFIX
            + evidence_block
            + EVIDENCE_SUFFIX
            + CONTINUATION_PROMPT
        )

        prefix_length = 0
        if (
            model_type == "vllm"
            and self.kv_cache_manager is not None
            and self.kv_cache_manager.is_available()
            and tokenizer is not None
        ):
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            kv_result = self.kv_cache_manager.prepare_continuation(
                prefix_tokens,
                evidence_block,
                CONTINUATION_PROMPT,
                tokenizer,
            )
            prefix_length = kv_result["prefix_length"]

        result = InjectionResult(
            augmented_input=augmented_input,
            prefix_length=prefix_length,
            compression_ratio=compression_ratio,
        )

        # Trigger speculative pre-fetching for the next retrieval
        if self.speculative_cache is not None:
            self.speculative_cache.on_retrieval_complete(evidence, prefix)

        return result
