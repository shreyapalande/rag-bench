# test_e2e.py
import os
from utils.chunker import DocumentProcessor
from retrievers.vector_retriever import VectorRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.hybrid_retriever import HybridRetriever
from generators.groq_generator import GroqGenerator
from generators.gemini_generator import GeminiGenerator

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
hybrid_retriever._is_setup = True
hybrid_retriever.chunks = chunks

# -------- GENERATORS --------
groq = GroqGenerator(model="llama-3.3-70b-versatile")
gemini = GeminiGenerator(model="gemini-2.5-flash")

retrievers = {
    "Vector": vector_retriever,
    "BM25": bm25_retriever,
    "Hybrid": hybrid_retriever
}

generators = {
    "LLaMA-3.3-70B": groq,
    "gemini-2.5-flash": gemini
}

# -------- TEST QUERIES --------
queries = [
    "What is the difference between BERT and GPT?",
    "How does key value caching improve transformer inference?",
    "What are the trade-offs between dense and sparse retrieval?"
]

print("=" * 70)
print("RAGBENCH — 3 RETRIEVERS × 2 MODELS")
print("=" * 70)

for query in queries:
    print(f"\nQUESTION: {query}")
    print("=" * 70)

    for ret_name, retriever in retrievers.items():
        ret_result = retriever.retrieve_and_time(query, top_k=3)

        for gen_name, generator in generators.items():
            gen_result = generator.generate_and_time(query, ret_result.chunks)

            print(f"\n[{ret_name} + {gen_name}]")
            print(f"Retrieve: {ret_result.latency_ms:.1f}ms | Generate: {gen_result.latency_ms:.0f}ms | Tokens: {gen_result.tokens_used}")
            print(gen_result.answer)
            print("-" * 70)