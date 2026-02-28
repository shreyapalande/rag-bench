# test_eval.py
# Research benchmark: rebuilds all retriever indexes from scratch
# to capture accurate setup metrics (time, memory, phase breakdown).
import json

from config import CONFIG
from utils.chunker import DocumentProcessor
from retrievers.vector_retriever import VectorRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.hybrid_retriever import HybridRetriever
from generators.groq_generator import GroqGenerator
from generators.gemini_generator import GeminiGenerator
from benchmark.runner import BenchmarkRunner
from benchmark.reporter import BenchmarkReporter

# ── Load chunks ───────────────────────────────────────────────────────
processor = DocumentProcessor()
print("Loading chunks...")
chunks = processor.load_chunks(CONFIG.data.chunks_path)
print(f"Loaded {len(chunks)} chunks\n")

# ── Setup retrievers (fresh builds → accurate setup metrics) ──────────
retrievers = {
    "Vector":  VectorRetriever(model_name=CONFIG.embedding.model_name),
    "BM25":    BM25Retriever(),
    "Hybrid":  HybridRetriever(rrf_k=CONFIG.benchmark.rrf_k),
}

print("Setting up retrievers...")
for name, ret in retrievers.items():
    print(f"\n  [{name}]")
    ret.setup_and_time(chunks)

# ── Generators ────────────────────────────────────────────────────────
generators = {
    "LLaMA":  GroqGenerator(model=CONFIG.api.groq_model),
    # "Gemini": GeminiGenerator(model=CONFIG.api.gemini_model),
}

# ── Ground truth ──────────────────────────────────────────────────────
with open(CONFIG.data.ground_truth_path) as f:
    gt_data = json.load(f)

questions    = [item["question"]    for item in gt_data]
ground_truth = {item["question"]: item["ground_truth"] for item in gt_data}

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
