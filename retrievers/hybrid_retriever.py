# retrievers/hybrid_retriever.py
import numpy as np
from retrievers.base import BaseRetriever
from retrievers.vector_retriever import VectorRetriever
from retrievers.bm25_retriever import BM25Retriever

class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining dense vector search and sparse BM25.
    Uses Reciprocal Rank Fusion (RRF) to merge results from both retrievers.
    
    RRF is more robust than weighted score averaging because:
    - Scores from different retrievers are not on the same scale
    - RRF only uses rank positions, not raw scores
    - No tuning required
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever = None,
        bm25_retriever: BM25Retriever = None,
        rrf_k: int = 60
    ):
        super().__init__(name="HybridRetriever")
        self.vector = vector_retriever or VectorRetriever()
        self.bm25 = bm25_retriever or BM25Retriever()
        self.rrf_k = rrf_k
        self.chunks = []

    def setup(self, chunks: list) -> None:
        """Setup both sub-retrievers and aggregate their phase metrics."""
        self.chunks = chunks

        print(f"[{self.name}] Setting up vector retriever...")
        self.vector.setup(chunks)

        print(f"[{self.name}] Setting up BM25 retriever...")
        self.bm25.setup(chunks)

        # Aggregate phase metrics from both sub-retrievers.
        # total_ms and memory_peak_mb are set by the outer setup_and_time().
        self.setup_metrics.embedding_ms = self.vector.setup_metrics.embedding_ms
        self.setup_metrics.indexing_ms = (
            self.vector.setup_metrics.indexing_ms + self.bm25.setup_metrics.indexing_ms
        )
        self.setup_metrics.tokenizing_ms = self.bm25.setup_metrics.tokenizing_ms

        print(f"[{self.name}] Hybrid setup complete")

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """
        Retrieve using Reciprocal Rank Fusion.
        
        RRF score for each chunk = sum of 1/(k + rank) across both retrievers.
        Higher score = better combined ranking.
        """
        # Get more candidates from each retriever than needed
        candidate_k = min(top_k * 3, len(self.chunks))

        vector_results = self.vector.retrieve(query, top_k=candidate_k)
        bm25_results = self.bm25.retrieve(query, top_k=candidate_k)

        # Build RRF score dictionary
        # Key: chunk text, Value: RRF score
        rrf_scores: dict[str, float] = {}

        for rank, chunk in enumerate(vector_results):
            rrf_scores[chunk] = rrf_scores.get(chunk, 0) + 1 / (self.rrf_k + rank + 1)

        for rank, chunk in enumerate(bm25_results):
            rrf_scores[chunk] = rrf_scores.get(chunk, 0) + 1 / (self.rrf_k + rank + 1)

        # Sort by RRF score descending
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [chunk for chunk, score in sorted_chunks[:top_k]]