"""
ColBERTv2 + PLAID/WARP Retrieval Engine

Wrapper for efficient late-interaction retrieval.
Integrates ColBERTv2 with PLAID and 
WARP optimizations.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ``colbert`` is a heavy GPU-leaning optional dep; it is imported lazily inside
# ``ColBERTRetriever.__init__`` so the module can be imported on CPU-only /
# API-only setups.


class ColBERTRetriever:
    """
    ColBERTv2 retriever with PLAID/WARP optimizations.
    
    Usage:
        retriever = ColBERTRetriever(
            index_path="data/indices/colbert/musique.plaid",
            checkpoint="colbert-ir/colbertv2.0"
        )
        
        results = retriever.retrieve("What year did the Berlin Wall fall?", k=5)
    """
    
    def __init__(
        self,
        index_path: str,
        checkpoint: str = "colbert-ir/colbertv2.0",
        k: int = 5,
        device: str = "cuda",
    ):
        from colbert import Searcher  # lazy: heavy optional dep
        from colbert.infra import ColBERTConfig, Run, RunConfig

        self.index_path = index_path
        self.checkpoint = checkpoint
        self.k = k
        self.device = device

        with Run().context(RunConfig(nranks=1, experiment="realm")):
            config = ColBERTConfig(
                query_maxlen=256,
                doc_maxlen=256,
                ncells=2,
                centroid_score_threshold=0.5,
                ndocs=16384,
            )
            self.searcher = Searcher(
                index=index_path,
                config=config,
            )
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        return_scores: bool = False,
    ) -> List[Dict]:
        """
        Retrieve top-k documents for query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve (default: self.k)
            return_scores: Whether to include retrieval scores
            
        Returns:
            List of retrieved documents with text and optionally scores
        """
        if k is None:
            k = self.k
        
        # Search
        results = self.searcher.search(query, k=k)
        
        # Format results
        formatted_results = []
        for passage_id, rank, score in zip(
            results[0], range(len(results[0])), results[1]
        ):
            doc = {
                "passage_id": passage_id,
                "text": self.searcher.collection[passage_id],
                "rank": rank,
            }
            if return_scores:
                doc["score"] = score
            formatted_results.append(doc)
        
        return formatted_results
    
    def retrieve_by_embedding(
        self,
        query_embedding: np.ndarray,
        fallback_text: str,
        k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve top-k documents using a dense query embedding from QueryGen.

        Uses ColBERT's internal scoring against the pre-computed document
        embeddings when available; falls back to text-based retrieval with
        *fallback_text* if the index does not support direct embedding lookup.

        Args:
            query_embedding: Dense query vector (e.g. 768-d from QueryGen).
            fallback_text: Text query used when embedding-based search is
                unavailable (ColBERT PLAID indices support text queries only
                in some configurations).
            k: Number of documents to retrieve (default: self.k).

        Returns:
            List of retrieved documents with text and optionally scores.
        """
        if k is None:
            k = self.k

        # Attempt embedding-based search via ColBERT's dense scoring path.
        # PLAID indices expose ``search_by_Q`` when the searcher is built
        # with decompressed centroid residuals.
        if hasattr(self.searcher, "search_by_Q"):
            import torch as _torch

            Q = _torch.tensor(query_embedding, dtype=_torch.float32)
            if Q.dim() == 1:
                Q = Q.unsqueeze(0)  # [1, dim]
            Q = Q.unsqueeze(0)  # [1, 1, dim] — batch of 1 query with 1 token repr
            results = self.searcher.search_by_Q(Q, k=k)
            formatted_results = []
            for passage_id, rank, score in zip(
                results[0], range(len(results[0])), results[1]
            ):
                formatted_results.append({
                    "passage_id": passage_id,
                    "text": self.searcher.collection[passage_id],
                    "rank": rank,
                    "score": float(score),
                })
            return formatted_results

        # Fallback: use text-based retrieval
        return self.retrieve(fallback_text, k=k)

    def retrieve_batch(
        self,
        queries: List[str],
        k: Optional[int] = None,
    ) -> List[List[Dict]]:
        """Batch retrieval for multiple queries."""
        results = []
        for query in queries:
            result = self.retrieve(query, k=k)
            results.append(result)
        return results
    
    def get_corpus_size(self) -> int:
        """Get number of documents in corpus."""
        return len(self.searcher.collection)
    
    def compute_embedding(self, text: str) -> torch.Tensor:
        """Compute query embedding (for policy network)."""
        # Use ColBERT's query encoder
        encoding = self.searcher.config.checkpoint.queryFromText(
            [text],
            bsize=1,
            with_ids=False,
        )
        return encoding[0].mean(dim=0)  # Average over tokens
