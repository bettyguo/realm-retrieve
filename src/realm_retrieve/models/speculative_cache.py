# SPDX-License-Identifier: Apache-2.0
"""
Speculative Caching System

After a retrieval, extract top-3 entities from retrieved content and
speculatively pre-fetch documents for each.  On the next retrieval
trigger, check whether the actual query matches a cached speculative
query (cosine similarity > 0.85).

Eliminates retrieval latency on cache hits.
"""

from __future__ import annotations

import atexit
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SpeculativeCache:
    """
    Thread-safe speculative caching layer around a retriever.

    After each retrieval the cache extracts entities from the returned
    passages, pre-fetches results for templated entity queries, and
    stores them keyed by query embedding.  Subsequent queries whose
    embedding is within *similarity_threshold* of a cached entry score
    a cache hit and skip the retriever entirely.
    """

    def __init__(
        self,
        retriever,
        encoder,
        similarity_threshold: float = 0.85,
        max_speculative: int = 3,
    ):
        self.retriever = retriever
        self.encoder = encoder
        self.similarity_threshold = similarity_threshold
        self.max_speculative = max_speculative

        self._lock = threading.Lock()
        self._cache: List[Tuple[np.ndarray, List[str]]] = []
        self._hits = 0
        self._misses = 0
        self.max_cache_size = 50

        # Persistent thread pool for non-blocking speculative pre-fetches
        self._executor = ThreadPoolExecutor(max_workers=4)
        atexit.register(self.shutdown)

    # ------------------------------------------------------------------
    # Entity extraction (spaCy, lazy-loaded)
    # ------------------------------------------------------------------

    _nlp = None

    @classmethod
    def _get_nlp(cls):
        if cls._nlp is None:
            import spacy
            try:
                cls._nlp = spacy.load("en_core_web_sm")
            except OSError:
                spacy.cli.download("en_core_web_sm")
                cls._nlp = spacy.load("en_core_web_sm")
        return cls._nlp

    def _extract_entities(self, text: str, top_k: int = 3) -> List[str]:
        nlp = self._get_nlp()
        doc = nlp(text)
        seen: dict[str, int] = {}
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"):
                seen[ent.text] = seen.get(ent.text, 0) + 1
        ranked = sorted(seen, key=seen.get, reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_retrieval_complete(
        self,
        retrieved_passages: List[str],
        topic: str,
    ) -> None:
        """Extract entities and launch speculative pre-fetches in parallel."""
        combined = " ".join(retrieved_passages)
        entities = self._extract_entities(combined, top_k=self.max_speculative)

        def _prefetch(entity):
            spec_query = f"What is {entity}'s role in {topic}?"
            spec_emb = self.encoder.encode(spec_query, convert_to_numpy=True)
            spec_emb = np.asarray(spec_emb, dtype=np.float32).squeeze()
            try:
                docs = self.retriever.retrieve(spec_query, k=5)
                passages = [d["text"] for d in docs]
            except Exception:
                return
            with self._lock:
                self._cache.append((spec_emb, passages))
                if len(self._cache) > self.max_cache_size:
                    self._cache = self._cache[-self.max_cache_size:]

        if entities:
            for entity in entities:
                self._executor.submit(_prefetch, entity)

    def check_cache(self, query_embedding: np.ndarray) -> Optional[List[str]]:
        """Return cached results on a hit, ``None`` on a miss."""
        q = np.asarray(query_embedding, dtype=np.float32).squeeze()
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            with self._lock:
                self._misses += 1
            return None

        with self._lock:
            for cached_emb, cached_passages in self._cache:
                c_norm = np.linalg.norm(cached_emb)
                if c_norm == 0:
                    continue
                sim = float(np.dot(q, cached_emb) / (q_norm * c_norm))
                if sim >= self.similarity_threshold:
                    self._hits += 1
                    return cached_passages
            self._misses += 1
        return None

    def clear(self) -> None:
        """Clear cache (call between questions)."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Return hit/miss statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }

    def shutdown(self) -> None:
        """Shut down the background executor, waiting for pending tasks."""
        self._executor.shutdown(wait=True)
