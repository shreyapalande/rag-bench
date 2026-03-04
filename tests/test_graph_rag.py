# tests/test_graph_rag.py
# Standalone test for GraphRAGRetriever.
# Run from project root: python tests/test_graph_rag.py
#
# Requires GROQ_API_KEY in .env
# No Redis needed.
#
# First run : extracts triples from all chunks (makes LLM API calls per chunk)
#             and saves graph to data/knowledge_graph.json
# Later runs: loads graph from disk instantly (no LLM calls)
#
# To force a fresh rebuild, delete data/knowledge_graph.json

from utils.chunker import DocumentProcessor
from retrievers.graph_rag_retriever import GraphRAGRetriever

CHUNKS_PATH = "data/processed_chunks.json"
TOP_K = 3

QUERIES = [
    "What is the difference between BERT and GPT?",
    "How does key value caching improve transformer inference?",
    "What are the trade-offs between dense and sparse retrieval?",
    "What is the attention mechanism in transformers?",
    "What is the difference between top-k and top-p sampling?",
]

# ── 1. Load chunks ────────────────────────────────────────────────────
print("Loading chunks...")
processor = DocumentProcessor()
chunks = processor.load_chunks(CHUNKS_PATH)
print(f"Loaded {len(chunks)} chunks\n")

# ── 2. Setup ──────────────────────────────────────────────────────────
# On first run this extracts triples from all chunks (LLM calls).
# On subsequent runs it loads from data/knowledge_graph.json instantly.
retriever = GraphRAGRetriever(max_hops=2)
retriever.setup_and_time(chunks)
print()

# ── 3. Graph statistics ───────────────────────────────────────────────
import networkx as nx
G = retriever._graph

entity_nodes = [n for n, d in G.nodes(data=True) if d.get("label") == "entity"]
chunk_nodes  = [n for n, d in G.nodes(data=True) if d.get("label") == "chunk"]

print("=" * 60)
print("GRAPH STATISTICS")
print("=" * 60)
print(f"  Total nodes    : {G.number_of_nodes()}")
print(f"    Entity nodes : {len(entity_nodes)}")
print(f"    Chunk nodes  : {len(chunk_nodes)}")
print(f"  Total edges    : {G.number_of_edges()}")
print(f"  Storage        : {retriever.setup_metrics.storage_mb:.2f} MB")
print(f"  Build time     : {retriever.setup_metrics.indexing_ms:.0f} ms")

print("\nTop 10 entities by connectivity:")
for entity, degree in retriever.top_entities(10):
    print(f"  [{degree:>3} edges]  {entity}")

# ── 4. Retrieval with trace ───────────────────────────────────────────
print("\n" + "=" * 60)
print("RETRIEVAL TRACES")
print("=" * 60)

for q in QUERIES:
    import time
    t0 = time.perf_counter()
    result = retriever.retrieve_and_time(q, top_k=TOP_K)
    latency = result.latency_ms

    seeds = retriever._seed_entities(q)
    chunk_ids = retriever._bfs_expand(seeds) if seeds else set()

    # Fallback seeds if empty
    if not chunk_ids:
        words = {w for w in q.lower().split() if len(w) > 3}
        broad_seeds = {
            n for n, d in G.nodes(data=True)
            if d.get("label") == "entity" and any(w in n for w in words)
        }
        seeds = broad_seeds
        chunk_ids = retriever._bfs_expand(broad_seeds) if broad_seeds else set()

    print(f"\n  Query    : {q}")
    print(f"  Seeds    : {sorted(seeds)[:6]}")
    print(f"  Chunks found via BFS : {len(chunk_ids)}")
    print(f"  Returned : {len(result.chunks)} chunks | Latency: {latency:.1f}ms")
    if result.chunks:
        print(f"  Chunk[0] : {result.chunks[0][:150]}...")

# ── 5. Latency summary ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LATENCY (5 queries)")
print("=" * 60)
times = []
for q in QUERIES:
    r = retriever.retrieve_and_time(q, top_k=TOP_K)
    times.append(r.latency_ms)
    print(f"  {q[:55]:<55}  {r.latency_ms:>6.1f}ms")
print(f"\n  Average: {sum(times)/len(times):.1f}ms")
print("\n  Note: GraphRAG latency is pure graph traversal (no LLM, no vector search).")
print("  Setup cost is paid once — LLM calls during triple extraction.")
