# tests/test_kv_cache.py
# Standalone test for KVCacheRetriever.
# Run from project root: python tests/test_kv_cache.py
#
# Requires Redis running locally:
#   docker run -p 6379:6379 redis
#   OR: redis-server

from utils.chunker import DocumentProcessor
from retrievers.kv_cache_retriever import KVCacheRetriever

CHUNKS_PATH = "data/processed_chunks.json"
TOP_K = 3

QUERIES = [
    "What is the difference between BERT and GPT?",
    "How does key value caching improve transformer inference?",
    "What are the trade-offs between dense and sparse retrieval?",
]

# ── 1. Load chunks ────────────────────────────────────────────────────
print("Loading chunks...")
processor = DocumentProcessor()
chunks = processor.load_chunks(CHUNKS_PATH)
print(f"Loaded {len(chunks)} chunks\n")

# ── 2. Setup ──────────────────────────────────────────────────────────
retriever = KVCacheRetriever(ttl=300)  # 5-min TTL for testing
retriever.setup_and_time(chunks)
print()

# ── 3. Cold run (all misses) ──────────────────────────────────────────
print("=" * 60)
print("COLD RUN — expecting all cache misses")
print("=" * 60)
for q in QUERIES:
    result = retriever.retrieve_and_time(q, top_k=TOP_K)
    hit = result.metadata["cache_hit"]
    print(f"\n  Query : {q[:55]}...")
    print(f"  Hit   : {hit}  |  Latency: {result.latency_ms:.1f}ms")
    print(f"  Chunk : {result.chunks[0][:120]}...")

print(f"\nAfter cold run — hits: {retriever.cache_hits}  misses: {retriever.cache_misses}  hit rate: {retriever.hit_rate:.0%}")

# ── 4. Warm run (all hits) ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("WARM RUN — expecting all cache hits")
print("=" * 60)
for q in QUERIES:
    result = retriever.retrieve_and_time(q, top_k=TOP_K)
    hit = result.metadata["cache_hit"]
    print(f"\n  Query : {q[:55]}...")
    print(f"  Hit   : {hit}  |  Latency: {result.latency_ms:.2f}ms")

print(f"\nAfter warm run — hits: {retriever.cache_hits}  misses: {retriever.cache_misses}  hit rate: {retriever.hit_rate:.0%}")

# ── 5. Latency comparison ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("LATENCY — MISS vs HIT (same query, back to back)")
print("=" * 60)

# Reset by using a fresh query not seen before
fresh_q = "What is the attention mechanism in transformers?"

# Miss
r1 = retriever.retrieve_and_time(fresh_q, top_k=TOP_K)
# Hit
r2 = retriever.retrieve_and_time(fresh_q, top_k=TOP_K)

print(f"\n  Query  : {fresh_q}")
print(f"  Miss   : {r1.latency_ms:.2f}ms  (vector search)")
print(f"  Hit    : {r2.latency_ms:.2f}ms  (Redis GET)")
if r1.latency_ms > 0:
    print(f"  Speedup: {r1.latency_ms / r2.latency_ms:.1f}x faster on cache hit")
