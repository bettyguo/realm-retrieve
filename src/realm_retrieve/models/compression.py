# SPDX-License-Identifier: Apache-2.0
"""
Implicit Compression Module

Attention-weighted sentence filtering for retrieved passages.
Retains only sentences whose relevance to the current reasoning query
exceeds the threshold tau_rel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class CompressedResult:
    """Output of the implicit compression step."""
    compressed_passages: List[str]
    compression_ratio: float
    utility_estimate: float


class ImplicitCompression:
    """
    Compress retrieved passages by retaining only query-relevant sentences.

    For each passage the module:
    1. Splits into sentences (nltk.sent_tokenize)
    2. Encodes each sentence with SBERT
    3. Scores via scaled-dot-product attention against the query embedding
    4. Retains sentences scoring above *tau_rel*
    5. Falls back to the single highest-scoring sentence when none pass

    """

    def __init__(
        self,
        tau_rel: float = 0.45,
        encoder_name: str = "all-MiniLM-L6-v2",
    ):
        self.tau_rel = tau_rel
        self.encoder_name = encoder_name
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.encoder_name)
        return self._encoder

    @staticmethod
    def _sent_tokenize(text: str) -> List[str]:
        try:
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        except Exception:
            import re
            return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def compress(
        self,
        passages: List[str],
        query_embedding: np.ndarray,
    ) -> CompressedResult:
        """
        Compress *passages* by retaining query-relevant sentences.

        Args:
            passages: raw retrieved passages
            query_embedding: [d] dense query vector

        Returns:
            CompressedResult with compressed passages, ratio, and utility.
        """
        if not passages:
            return CompressedResult(
                compressed_passages=[],
                compression_ratio=0.0,
                utility_estimate=1.0,
            )

        encoder = self._get_encoder()
        query_emb = np.asarray(query_embedding, dtype=np.float32)
        if query_emb.ndim > 1:
            query_emb = query_emb.squeeze()

        total_chars = 0
        retained_chars = 0
        compressed_passages: List[str] = []
        all_full_embs: List[np.ndarray] = []
        all_compressed_embs: List[np.ndarray] = []

        for passage in passages:
            sentences = self._sent_tokenize(passage)
            if not sentences:
                compressed_passages.append(passage)
                total_chars += len(passage)
                retained_chars += len(passage)
                continue

            total_chars += sum(len(s) for s in sentences)

            sent_embeddings = encoder.encode(sentences, convert_to_numpy=True)
            sent_embeddings = np.asarray(sent_embeddings, dtype=np.float32)

            d = query_emb.shape[-1]
            scores = sent_embeddings @ query_emb / np.sqrt(d)

            # Sigmoid scoring: each sentence gets an independent relevance
            # score in [0, 1] so that tau_rel acts as an absolute threshold
            # rather than requiring a sentence to dominate the softmax mass.
            attention_weights = 1.0 / (1.0 + np.exp(-scores))

            mask = attention_weights > self.tau_rel
            if not mask.any():
                mask[np.argmax(attention_weights)] = True

            retained = [s for s, m in zip(sentences, mask) if m]
            compressed_passages.append(" ".join(retained))
            retained_chars += sum(len(s) for s in retained)

            full_emb = sent_embeddings.mean(axis=0)
            comp_emb = sent_embeddings[mask].mean(axis=0)
            all_full_embs.append(full_emb)
            all_compressed_embs.append(comp_emb)

        compression_ratio = 1.0 - (retained_chars / total_chars) if total_chars > 0 else 0.0

        if all_full_embs and all_compressed_embs:
            full_agg = np.mean(all_full_embs, axis=0)
            comp_agg = np.mean(all_compressed_embs, axis=0)
            cos_sim = float(
                np.dot(full_agg, comp_agg)
                / (np.linalg.norm(full_agg) * np.linalg.norm(comp_agg) + 1e-10)
            )
            utility_estimate = max(0.0, min(1.0, cos_sim))
        else:
            utility_estimate = 1.0

        return CompressedResult(
            compressed_passages=compressed_passages,
            compression_ratio=compression_ratio,
            utility_estimate=utility_estimate,
        )
