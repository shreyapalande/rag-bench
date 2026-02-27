# test_e2e.py
import os
from utils.chunker import DocumentProcessor
from retrievers.vector_retriever import VectorRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.hybrid_retriever import HybridRetriever
from generators.groq_generator import GroqGenerator

# -------- LOAD --------
processor = DocumentProcessor()
chunks = processor.load_chunks("data/processed_chunks.json")

vector_retriever = VectorRetriever()
vector_retriever.load("data/faiss_index.bin", "data/chunks.pkl")

bm25_retriever = BM25Retriever()
bm25_retriever.setup_and_time(chunks)

hybrid_retriever = HybridRetriever(
    vector_retriever=vector_retriever,
    bm25_retriever=bm25_retriever
)
# Mark as already setup — no need to re-embed
hybrid_retriever._is_setup = True
hybrid_retriever.chunks = chunks

generator = GroqGenerator(model="llama-3.3-70b-versatile")

# -------- TEST QUERIES --------
queries = [
    "What is the difference between BERT and GPT?",
    "How does key value caching improve transformer inference?",
    "What are the trade-offs between dense and sparse retrieval?"
]

print("=" * 70)
print("FULL RAG PIPELINE — VECTOR vs BM25 vs HYBRID")
print("=" * 70)

for query in queries:
    print(f"\nQUESTION: {query}")
    print("-" * 70)

    vec_chunks = vector_retriever.retrieve_and_time(query, top_k=3)
    bm25_chunks = bm25_retriever.retrieve_and_time(query, top_k=3)
    hybrid_chunks = hybrid_retriever.retrieve_and_time(query, top_k=3)

    vec_answer = generator.generate_and_time(query, vec_chunks.chunks)
    bm25_answer = generator.generate_and_time(query, bm25_chunks.chunks)
    hybrid_answer = generator.generate_and_time(query, hybrid_chunks.chunks)

    print(f"\nVECTOR ({vec_chunks.latency_ms:.1f}ms retrieve | {vec_answer.latency_ms:.0f}ms generate | {vec_answer.tokens_used} tokens):")
    print(vec_answer.answer)

    print(f"\nBM25 ({bm25_chunks.latency_ms:.1f}ms retrieve | {bm25_answer.latency_ms:.0f}ms generate | {bm25_answer.tokens_used} tokens):")
    print(bm25_answer.answer)

    print(f"\nHYBRID ({hybrid_chunks.latency_ms:.1f}ms retrieve | {hybrid_answer.latency_ms:.0f}ms generate | {hybrid_answer.tokens_used} tokens):")
    print(hybrid_answer.answer)

    print("=" * 70)