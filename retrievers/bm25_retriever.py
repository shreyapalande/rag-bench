# retrievers/bm25_retriever.py
import re
from rank_bm25 import BM25Okapi
from retrievers.base import BaseRetriever

class BM25Retriever(BaseRetriever):
    """
    Sparse keyword retriever using BM25Okapi.
    No embeddings needed — pure term frequency matching.
    Fast setup, good for exact keyword and technical term matching.
    """

    def __init__(self):
        super().__init__(name="BM25Retriever")
        self.bm25 = None
        self.chunks = []

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase and split on non-alphanumeric characters."""
        text = text.lower()
        tokens = re.split(r"[^a-z0-9]+", text)
        return [t for t in tokens if t]

    def setup(self, chunks: list) -> None:
        """Tokenize all chunks and build BM25 index."""
        self.chunks = chunks
        print(f"[{self.name}] Tokenizing {len(chunks)} chunks...")
        tokenized = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"[{self.name}] BM25 index built")

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Return top_k chunks by BM25 score."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices sorted by score descending
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [self.chunks[i].text for i in top_indices]