# test_eval.py
import json
import os
from utils.chunker import DocumentProcessor
from retrievers.vector_retriever import VectorRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.hybrid_retriever import HybridRetriever
from generators.groq_generator import GroqGenerator
from generators.gemini_generator import GeminiGenerator
from evaluation.llm_judge import LLMJudge

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

groq = GroqGenerator(model="llama-3.3-70b-versatile")
gemini = GeminiGenerator(model="gemini-2.0-flash")
judge = LLMJudge(model="llama-3.3-70b-versatile")

# -------- LOAD GROUND TRUTH --------
with open("data/ground_truth.json", "r") as f:
    ground_truth_data = json.load(f)

questions = [item["question"] for item in ground_truth_data]
ground_truth_map = {
    item["question"]: item["ground_truth"]
    for item in ground_truth_data
}

retrievers = {
    "Vector": vector_retriever,
    "BM25": bm25_retriever,
    "Hybrid": hybrid_retriever
}

generators = {
    "LLaMA": groq,
    # "Gemini": gemini
}

# -------- RUN BENCHMARK --------
all_results = {}
report_rows = []

for ret_name, retriever in retrievers.items():
    for gen_name, generator in generators.items():
        combo = f"{ret_name}+{gen_name}"
        print(f"\nEvaluating {combo}...")

        samples = []
        latencies = []

        for question in questions:
            import time

            # Retrieve
            ret_result = retriever.retrieve_and_time(question, top_k=3)

            # Generate
            gen_result = generator.generate_and_time(question, ret_result.chunks)

            total_latency = ret_result.latency_ms + gen_result.latency_ms
            latencies.append(total_latency)

            samples.append({
                "question": question,
                "answer": gen_result.answer,
                "contexts": ret_result.chunks,
                "ground_truth": ground_truth_map[question]
            })

        # Judge
        avg_score, _ = judge.evaluate_batch(samples)
        avg_latency = sum(latencies) / len(latencies)

        all_results[combo] = {
            "scores": avg_score.to_dict(),
            "avg_latency_ms": round(avg_latency, 1)
        }

        row = {
            "combo": combo,
            **avg_score.to_dict(),
            "avg_latency_ms": round(avg_latency, 1)
        }
        report_rows.append(row)
        print(f"  Done — Average: {avg_score.average():.3f} | Latency: {avg_latency:.0f}ms")

# -------- PRINT SUMMARY TABLE --------
print("\n" + "=" * 90)
print("RAGBENCH RESULTS — 3 RETRIEVERS × 2 MODELS")
print("=" * 90)
print(f"{'Combo':<20} {'Faith':>7} {'Relevancy':>10} {'Ctx Rel':>8} {'Complete':>9} {'Avg':>6} {'Latency':>9}")
print("-" * 90)

for row in sorted(report_rows, key=lambda x: x["average"], reverse=True):
    print(
        f"{row['combo']:<20} "
        f"{row['faithfulness']:>7.3f} "
        f"{row['answer_relevancy']:>10.3f} "
        f"{row['context_relevancy']:>8.3f} "
        f"{row['completeness']:>9.3f} "
        f"{row['average']:>6.3f} "
        f"{row['avg_latency_ms']:>8.0f}ms"
    )

# -------- SAVE RESULTS --------
os.makedirs("reports", exist_ok=True)
with open("reports/benchmark_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\nResults saved to reports/benchmark_results.json")