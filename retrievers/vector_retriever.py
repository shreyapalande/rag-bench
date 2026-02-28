# retrievers/vector_retriever.py
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from retrievers.base import BaseRetriever

class VectorRetriever(BaseRetriever):
    """
    Dense vector retriever using FAISS + sentence-transformers.
    Uses cosine similarity (IndexFlatIP with normalized embeddings).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(name="VectorRetriever")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def setup(self, chunks: list) -> None:
        """Embed all chunks and build FAISS index, recording per-phase timing."""
        import time
        self.chunks = chunks
        texts = [c.text for c in chunks]

        # Phase 1: embedding
        print(f"[{self.name}] Embedding {len(texts)} chunks...")
        t0 = time.perf_counter()
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = np.array(embeddings).astype("float32")
        self.setup_metrics.embedding_ms = (time.perf_counter() - t0) * 1000

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Phase 2: index build
        t0 = time.perf_counter()
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.setup_metrics.indexing_ms = (time.perf_counter() - t0) * 1000

        print(f"[{self.name}] Indexed {len(texts)} chunks")

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Return top_k chunks by cosine similarity."""
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        return [self.chunks[i].text for i in indices[0] if i < len(self.chunks)]

    def save(self, index_path: str, chunks_path: str):
        """Persist index and chunks to disk."""
        faiss.write_index(self.index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, index_path: str, chunks_path: str):
        """Load index and chunks from disk."""
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        self._is_setup = True