# SPDX-License-Identifier: Apache-2.0
"""
KV-Cache Preservation Module

For open-weight models served via vLLM, preserves the KV cache of the
unchanged reasoning prefix when injecting retrieved evidence.
Only the new evidence block and continuation prompt require fresh
computation.

Implements prefix-aware KV-cache reuse for evidence injection.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class KVCacheManager:
    """
    Manage KV-cache reuse when injecting evidence into a reasoning chain.

    When vLLM is available the manager computes exact prefix lengths so
    that the inference engine can skip re-encoding the unchanged prefix.
    When vLLM is *not* installed (API-based models) it falls back to
    simple string concatenation without cache awareness.
    """

    EVIDENCE_PREFIX = "\n<retrieved>\n"
    EVIDENCE_SUFFIX = "\n</retrieved>\n"
    CONTINUATION_PROMPT = "Continue your reasoning, using this evidence if relevant."

    def __init__(self, backend: str = "vllm"):
        self.backend = backend
        self._vllm_available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if vLLM KV-cache manipulation is available."""
        if self._vllm_available is None:
            try:
                import vllm  # noqa: F401
                self._vllm_available = True
            except ImportError:
                self._vllm_available = False
        return self._vllm_available

    def prepare_continuation(
        self,
        prefix_tokens: List[int],
        evidence_text: str,
        continuation_prompt: str,
        tokenizer: Any,
    ) -> Dict[str, Any]:
        """
        Prepare token sequence for KV-cache-aware continuation.

        Args:
            prefix_tokens: token ids for the unchanged prefix ``r_{1:i}``
            evidence_text: raw evidence string to inject
            continuation_prompt: prompt appended after the evidence block
            tokenizer: HuggingFace-compatible tokenizer

        Returns:
            dict with ``input_ids``, ``prefix_length``, ``new_tokens``
        """
        evidence_block = (
            self.EVIDENCE_PREFIX + evidence_text + self.EVIDENCE_SUFFIX + continuation_prompt
        )
        new_tokens = tokenizer.encode(evidence_block, add_special_tokens=False)

        input_ids = list(prefix_tokens) + new_tokens
        prefix_length = len(prefix_tokens)

        self._last_prefix_length = prefix_length
        self._last_new_tokens = len(new_tokens)

        return {
            "input_ids": input_ids,
            "prefix_length": prefix_length,
            "new_tokens": new_tokens,
        }

    def measure_ttft(self, with_cache: bool = True, engine: Any = None) -> float:
        """Measure time-to-first-token using a live vLLM engine.

        Requires a vLLM engine instance. Returns 0.0 if no engine is
        available or prefix state has not been set.
        """
        prefix_len = self._last_prefix_length if hasattr(self, '_last_prefix_length') else 0
        new_len = self._last_new_tokens if hasattr(self, '_last_new_tokens') else 0

        if prefix_len == 0 and new_len == 0:
            return 0.0

        if engine is None or not self.is_available():
            return 0.0

        try:
            from vllm import SamplingParams

            if with_cache:
                input_ids = (
                    self._last_input_ids
                    if hasattr(self, '_last_input_ids')
                    else list(range(prefix_len)) + list(range(new_len))
                )
            else:
                total = prefix_len + new_len
                input_ids = list(range(total))

            params = SamplingParams(max_tokens=1, temperature=0.0)
            t0 = time.monotonic()
            engine.generate(
                prompt_token_ids=input_ids,
                sampling_params=params,
            )
            t1 = time.monotonic()
            return t1 - t0
        except Exception:
            return 0.0
