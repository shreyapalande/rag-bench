# retrievers/kv_cache_retriever.py
import hashlib
import json
import os
import time

import redis
from dotenv import load_dotenv

from retrievers.base import BaseRetriever, RetrievalResult
from retrievers.vector_retriever import VectorRetriever

load_dotenv()


class KVCacheRetriever(BaseRetriever):
    """
    KV Cache retriever: wraps VectorRetriever with Redis exact-match caching.

    On cache HIT  → returns cached chunks instantly (sub-millisecond Redis GET)
    On cache MISS → runs FAISS vector search, stores chunks in Redis with TTL

    Cache key   : kv_cache::{sha256(normalized_query)[:16]}
    Cache value : JSON-serialised list of chunk texts

    Useful when identical (or near-identical) queries repeat across sessions.
    Run warm_cache() before benchmarking to pre-populate the cache and
    demonstrate hit-path latency.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        redis_url: str = None,
        ttl: int = 3600,
    ):
        super().__init__(name="KVCacheRetriever")
        self.ttl = ttl
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._vector = VectorRetriever(model_name=model_name)
        self._redis: redis.Redis = None
        self.cache_hits = 0
        self.cache_misses = 0

    # ── Helpers ───────────────────────────────────────────────────────

    def _cache_key(self, query: str) -> str:
        normalized = query.strip().lower()
        digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return f"kv_cache::{digest}"

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    # ── BaseRetriever interface ────────────────────────────────────────

    def setup(self, chunks: list) -> None:
        """Build FAISS vector index and connect to Redis, clearing stale cache."""
        # Phase 1: vector index — inherits embedding_ms + indexing_ms
        self._vector.setup(chunks)
        self.setup_metrics.embedding_ms = self._vector.setup_metrics.embedding_ms
        self.setup_metrics.indexing_ms = self._vector.setup_metrics.indexing_ms

        # Phase 2: Redis connection + stale cache flush
        self._redis = redis.from_url(self.redis_url, decode_responses=True)
        self._redis.ping()
        stale = self._redis.keys("kv_cache::*")
        if stale:
            self._redis.delete(*stale)
            print(f"[{self.name}] Cleared {len(stale)} stale cache entries")
        print(f"[{self.name}] Redis connected — TTL={self.ttl}s")

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        key = self._cache_key(query)
        cached = self._redis.get(key)
        if cached:
            self.cache_hits += 1
            return json.loads(cached)

        # Cache miss: vector search → store result
        self.cache_misses += 1
        chunks = self._vector.retrieve(query, top_k=top_k)
        self._redis.setex(key, self.ttl, json.dumps(chunks))
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

        Call this before benchmarking so that those queries hit the cache
        during the timed benchmark run, demonstrating hit-path latency.
        """
        print(f"[{self.name}] Warming cache with {len(questions)} queries...")
        for q in questions:
            self.retrieve(q, top_k=top_k)
        # Reset counters so warm-up isn't counted in benchmark stats
        self.cache_hits = 0
        self.cache_misses = 0
        print(f"[{self.name}] Cache warmed — counters reset")
