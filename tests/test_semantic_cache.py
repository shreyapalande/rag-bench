# tests/test_semantic_cache.py
# Standalone test for SemanticCacheRetriever.
# Run from project root: python tests/test_semantic_cache.py
#
# Requires Redis running locally:
#   docker run -p 6379:6379 redis
#   OR: redis-server

from utils.chunker import DocumentProcessor
from retrievers.semantic_cache_retriever import SemanticCacheRetriever

CHUNKS_PATH = "data/processed_chunks.json"
TOP_K = 3

QUERIES = [
    "What is the difference between BERT and GPT?",
    "How does key value caching improve transformer inference?",
    "What are the trade-offs between dense and sparse retrieval?",
]

# Paraphrased versions — semantic cache should HIT; KV Cache would MISS these
PARAPHRASED = [
    "How do BERT and GPT differ from each other?",
    "In what way does KV caching speed up transformer models?",
    "What are the pros and cons of dense versus sparse retrieval?",
]

# ── 1. Load chunks ────────────────────────────────────────────────────
print("Loading chunks...")
processor = DocumentProcessor()
chunks = processor.load_chunks(CHUNKS_PATH)
print(f"Loaded {len(chunks)} chunks\n")

# ── 2. Setup ──────────────────────────────────────────────────────────
retriever = SemanticCacheRetriever(ttl=300, similarity_threshold=0.85)
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

# ── 4. Exact warm run (all hits) ──────────────────────────────────────
print("\n" + "=" * 60)
print("EXACT WARM RUN — expecting all cache hits")
print("=" * 60)
for q in QUERIES:
    result = retriever.retrieve_and_time(q, top_k=TOP_K)
    hit = result.metadata["cache_hit"]
    print(f"\n  Query : {q[:55]}...")
    print(f"  Hit   : {hit}  |  Latency: {result.latency_ms:.2f}ms")

print(f"\nAfter exact warm run — hits: {retriever.cache_hits}  misses: {retriever.cache_misses}  hit rate: {retriever.hit_rate:.0%}")

# ── 5. Paraphrase test — the key differentiator vs KV Cache ───────────
print("\n" + "=" * 60)
print("PARAPHRASE TEST — differently worded but same-meaning queries")
print("Semantic cache should HIT; KV Cache would MISS these")
print("=" * 60)
for orig, para in zip(QUERIES, PARAPHRASED):
    result = retriever.retrieve_and_time(para, top_k=TOP_K)
    hit = result.metadata["cache_hit"]
    print(f"\n  Original   : {orig[:55]}...")
    print(f"  Paraphrase : {para[:55]}...")
    print(f"  Hit        : {hit}  |  Latency: {result.latency_ms:.2f}ms")

print(f"\nAfter paraphrase run — hits: {retriever.cache_hits}  misses: {retriever.cache_misses}  hit rate: {retriever.hit_rate:.0%}")

# ── 6. Latency comparison: MISS vs HIT ────────────────────────────────
print("\n" + "=" * 60)
print("LATENCY — MISS vs HIT (same query, back to back)")
print("=" * 60)

fresh_q = "What is the attention mechanism in transformers?"

# Miss
r1 = retriever.retrieve_and_time(fresh_q, top_k=TOP_K)
# Hit
r2 = retriever.retrieve_and_time(fresh_q, top_k=TOP_K)

print(f"\n  Query  : {fresh_q}")
print(f"  Miss   : {r1.latency_ms:.2f}ms  (embed + FAISS search)")
print(f"  Hit    : {r2.latency_ms:.2f}ms  (embed + Redis scan + cosine sim)")
if r1.latency_ms > 0:
    print(f"  Speedup: {r1.latency_ms / r2.latency_ms:.1f}x faster on cache hit")

# ── 7. Threshold sensitivity ──────────────────────────────────────────
print("\n" + "=" * 60)
print("THRESHOLD NOTE")
print("=" * 60)
print(f"  Current threshold : {retriever.similarity_threshold}")
print("  Lower threshold (e.g. 0.75) → more hits, may return less relevant cached chunks")
print("  Higher threshold (e.g. 0.95) → fewer hits, only near-identical queries cached")
