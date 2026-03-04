# retrievers/semantic_cache_retriever.py
import json
import os
import time
import uuid

import numpy as np
import redis
from dotenv import load_dotenv

from retrievers.base import BaseRetriever, RetrievalResult
from retrievers.vector_retriever import VectorRetriever

load_dotenv()


class SemanticCacheRetriever(BaseRetriever):
    """
    Semantic Cache retriever: wraps VectorRetriever with Redis semantic caching.

    Unlike KVCacheRetriever (exact-match), this caches query embeddings and
    returns cached results when a new query is semantically similar
    (cosine similarity >= threshold) to a previously-seen query.

    Cache key   : sem_cache::{uuid12}
    Cache value : JSON {"embedding": [...], "chunks": [...]}

    On cache HIT  → returns cached chunks (embed + scan + cosine similarity)
    On cache MISS → runs FAISS vector search, stores embedding+chunks in Redis

    Key advantage over KV Cache: catches paraphrased / near-duplicate queries
    that differ in wording but ask the same thing.

    Run warm_cache() before benchmarking to pre-populate the cache and
    demonstrate hit-path latency.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        redis_url: str = None,
        ttl: int = 3600,
        similarity_threshold: float = 0.85,
    ):
        super().__init__(name="SemanticCacheRetriever")
        self.ttl = ttl
        self.similarity_threshold = similarity_threshold
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._vector = VectorRetriever(model_name=model_name)
        self._model = self._vector.model  # reuse loaded model — no double load
        self._redis: redis.Redis = None
        self.cache_hits = 0
        self.cache_misses = 0

    # ── Helpers ───────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed and L2-normalise a single query. Returns shape (1, dim)."""
        emb = self._model.encode([query])
        emb = np.array(emb, dtype="float32")
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.maximum(norms, 1e-10)

    def _find_similar(self, query_emb: np.ndarray) -> list[str] | None:
        """
        Scan all cached entries; return chunks for the most similar cached query
        if cosine similarity >= threshold, else None.

        Cosine similarity = dot product of L2-normalised vectors.
        For small caches (< 1000 entries) linear scan is negligible.
        """
        keys = self._redis.keys("sem_cache::*")
        if not keys:
            return None

        best_sim = -1.0
        best_chunks = None

        for key in keys:
            raw = self._redis.get(key)
            if not raw:
                continue
            entry = json.loads(raw)
            cached_emb = np.array(entry["embedding"], dtype="float32")
            sim = float(np.dot(query_emb[0], cached_emb))
            if sim > best_sim:
                best_sim = sim
                best_chunks = entry["chunks"]

        return best_chunks if best_sim >= self.similarity_threshold else None

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    # ── BaseRetriever interface ────────────────────────────────────────

    def setup(self, chunks: list) -> None:
        """Build FAISS vector index and connect to Redis, clearing stale cache."""
        # Phase 1: vector index — inherit embedding_ms + indexing_ms
        self._vector.setup(chunks)
        self.setup_metrics.embedding_ms = self._vector.setup_metrics.embedding_ms
        self.setup_metrics.indexing_ms = self._vector.setup_metrics.indexing_ms

        # Phase 2: Redis connection + stale cache flush
        self._redis = redis.from_url(self.redis_url, decode_responses=True)
        self._redis.ping()
        stale = self._redis.keys("sem_cache::*")
        if stale:
            self._redis.delete(*stale)
            print(f"[{self.name}] Cleared {len(stale)} stale cache entries")
        print(
            f"[{self.name}] Redis connected — "
            f"TTL={self.ttl}s | threshold={self.similarity_threshold}"
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        query_emb = self._embed_query(query)

        # Check semantic cache first
        cached = self._find_similar(query_emb)
        if cached is not None:
            self.cache_hits += 1
            return cached

        # Cache miss: FAISS vector search → store embedding + chunks
        self.cache_misses += 1
        chunks = self._vector.retrieve(query, top_k=top_k)

        entry = {"embedding": query_emb[0].tolist(), "chunks": chunks}
        key = f"sem_cache::{uuid.uuid4().hex[:12]}"
        self._redis.setex(key, self.ttl, json.dumps(entry))
        return chunks

    def retrieve_and_time(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Override to include cache hit/miss in metadata."""
        if not self._is_setup:
            raise RuntimeError(f"[{self.name}] Call setup() before retrieve()")
        hits_before = self.cache_hits
        t0 = time.perf_counter()
        chunks = self.retrieve(query, top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        cache_hit = self.cache_hits > hits_before
        total = self.cache_hits + self.cache_misses
        return RetrievalResult(
            chunks=chunks,
            latency_ms=latency_ms,
            metadata={
                "retriever": self.name,
                "top_k": top_k,
                "cache_hit": cache_hit,
                "hit_rate": round(self.cache_hits / total, 3) if total > 0 else 0.0,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
            },
        )

    # ── Cache warming ─────────────────────────────────────────────────

    def warm_cache(self, questions: list[str], top_k: int = 5) -> None:
        """
        Pre-populate the cache with a list of questions.

        Call this before benchmarking so those queries hit the cache
        during the timed run, demonstrating hit-path latency.
        """
        print(f"[{self.name}] Warming cache with {len(questions)} queries...")
        for q in questions:
            self.retrieve(q, top_k=top_k)
        # Reset counters so warm-up isn't counted in benchmark stats
        self.cache_hits = 0
        self.cache_misses = 0
        print(f"[{self.name}] Cache warmed — counters reset")
