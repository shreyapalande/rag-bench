# test_eval.py
# Research benchmark: rebuilds all retriever indexes from scratch
# to capture accurate setup metrics (time, memory, phase breakdown).
import json

from config import CONFIG
from utils.chunker import DocumentProcessor
from retrievers.vector_retriever import VectorRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.kv_cache_retriever import KVCacheRetriever
from retrievers.semantic_cache_retriever import SemanticCacheRetriever
from generators.groq_generator import GroqGenerator
from generators.gemini_generator import GeminiGenerator
from benchmark.runner import BenchmarkRunner
from benchmark.reporter import BenchmarkReporter

# ── Load chunks ───────────────────────────────────────────────────────
processor = DocumentProcessor()
print("Loading chunks...")
chunks = processor.load_chunks(CONFIG.data.chunks_path)
print(f"Loaded {len(chunks)} chunks\n")

# ── Load ground truth (needed before cache warm-up) ───────────────────
with open(CONFIG.data.ground_truth_path) as f:
    gt_data = json.load(f)

questions    = [item["question"]    for item in gt_data]
ground_truth = {item["question"]: item["ground_truth"] for item in gt_data}

# ── Setup retrievers (fresh builds → accurate setup metrics) ──────────
kv_cache = KVCacheRetriever(ttl=3600)
sem_cache = SemanticCacheRetriever(ttl=3600, similarity_threshold=0.85)

retrievers = {
    "Vector":    VectorRetriever(model_name=CONFIG.embedding.model_name),
    "BM25":      BM25Retriever(),
    "Hybrid":    HybridRetriever(rrf_k=CONFIG.benchmark.rrf_k),
    "KVCache":   kv_cache,
    "SemCache":  sem_cache,
}

print("Setting up retrievers...")
for name, ret in retrievers.items():
    print(f"\n  [{name}]")
    ret.setup_and_time(chunks)

# ── Warm cache retrievers with first half of questions ────────────────
# Pre-populates Redis so those queries are cache HITs in the benchmark,
# demonstrating hit-path latency vs miss-path (remaining questions).
warm_questions = questions[: len(questions) // 2]
kv_cache.warm_cache(warm_questions, top_k=CONFIG.benchmark.top_k)
sem_cache.warm_cache(warm_questions, top_k=CONFIG.benchmark.top_k)

# ── Generators ────────────────────────────────────────────────────────
generators = {
    "LLaMA":  GroqGenerator(model=CONFIG.api.groq_model),
    # "Gemini": GeminiGenerator(model=CONFIG.api.gemini_model),
}

# ── Run benchmark ─────────────────────────────────────────────────────
runner = BenchmarkRunner(
    retrievers=retrievers,
    generators=generators,
    questions=questions,
    ground_truth=ground_truth,
    top_k=CONFIG.benchmark.top_k,
)
results = runner.run_all()

# ── Save reports ──────────────────────────────────────────────────────
reporter = BenchmarkReporter(output_dir=CONFIG.benchmark.output_dir)
reporter.save_all(results)
