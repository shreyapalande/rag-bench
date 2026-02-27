# test_setup.py
import os
from utils.chunker import DocumentProcessor
from retrievers.vector_retriever import VectorRetriever

CHUNKS_PATH = "data/processed_chunks.json"
INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PKL_PATH = "data/chunks.pkl"

processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
retriever = VectorRetriever()

# Only re-process and re-embed if files don't exist
if not os.path.exists(CHUNKS_PATH):
    print("Processing PDFs...")
    chunks = processor.process_directory("data/")
    processor.save_chunks(chunks, CHUNKS_PATH)
else:
    print("Loading existing chunks...")
    chunks = processor.load_chunks(CHUNKS_PATH)

if not os.path.exists(INDEX_PATH):
    print("Building FAISS index...")
    retriever.setup_and_time(chunks)
    retriever.save(INDEX_PATH, CHUNKS_PKL_PATH)
else:
    print("Loading existing index...")
    retriever.load(INDEX_PATH, CHUNKS_PKL_PATH)

# Test query
result = retriever.retrieve_and_time(
    "What are the trade-offs between dense and sparse retrieval?",
    top_k=3
)

print(f"\nLatency: {result.latency_ms:.1f}ms")
print(f"\nTop chunk:\n{result.chunks[0][:300]}")

# add to test_setup.py temporarily
print("\n--- CHUNK VERIFICATION ---")
print(f"Total chunks: {len(chunks)}")
print(f"Avg chunk length: {sum(len(c.text) for c in chunks) / len(chunks):.0f} chars")
print(f"\nChunk 0 from {chunks[0].file_name}:")
print(chunks[0].text[:400])
print(f"\nChunk 10 from {chunks[10].file_name}:")
print(chunks[10].text[:400])

# Add at bottom of test_setup.py
from retrievers.bm25_retriever import BM25Retriever

print("\n--- BM25 vs VECTOR COMPARISON ---")

bm25_retriever = BM25Retriever()
bm25_retriever.setup_and_time(chunks)

comparison_queries = [
    "What is the attention mechanism?",
    "KV cache transformer inference",
    "BERT GPT difference bidirectional"
]

for query in comparison_queries:
    vec_result = retriever.retrieve_and_time(query, top_k=1)
    bm25_result = bm25_retriever.retrieve_and_time(query, top_k=1)

    print(f"\nQuery: {query}")
    print(f"Vector ({vec_result.latency_ms:.1f}ms): {vec_result.chunks[0][:150]}")
    print(f"BM25   ({bm25_result.latency_ms:.1f}ms): {bm25_result.chunks[0][:150]}")
    print("-" * 60)