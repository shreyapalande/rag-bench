# benchmark/runner.py
from dataclasses import dataclass, field

from retrievers.base import BaseRetriever
from generators.base import BaseGenerator
from utils.profiler import SetupMetrics
from evaluation.llm_judge import LLMJudge


@dataclass
class ComboResult:
    """All metrics for one retriever × generator combination."""
    retriever_name: str
    generator_name: str
    setup_metrics: SetupMetrics
    avg_retrieval_ms: float
    avg_generation_ms: float
    avg_total_ms: float
    avg_tokens: float
    judge_scores: dict
    per_question: list[dict] = field(default_factory=list)

    @property
    def combo(self) -> str:
        return f"{self.retriever_name}+{self.generator_name}"

    def summary_row(self) -> dict:
        """Flat dict suitable for CSV export."""
        return {
            "combo": self.combo,
            "retriever": self.retriever_name,
            "generator": self.generator_name,
            # setup cost
            "setup_total_ms": round(self.setup_metrics.total_ms, 1),
            "setup_embedding_ms": round(self.setup_metrics.embedding_ms, 1),
            "setup_tokenizing_ms": round(self.setup_metrics.tokenizing_ms, 1),
            "setup_indexing_ms": round(self.setup_metrics.indexing_ms, 1),
            "setup_memory_peak_mb": round(self.setup_metrics.memory_peak_mb, 2),
            # query latency
            "avg_retrieval_ms": round(self.avg_retrieval_ms, 1),
            "avg_generation_ms": round(self.avg_generation_ms, 1),
            "avg_total_ms": round(self.avg_total_ms, 1),
            "avg_tokens": round(self.avg_tokens, 1),
            # quality
            **{k: round(v, 3) for k, v in self.judge_scores.items()},
        }


class BenchmarkRunner:
    """
    Orchestrates benchmarks across all retriever × generator combinations.

    For each combination it:
      1. Runs every question through retrieve → generate
      2. Collects per-query latency and token counts
      3. Evaluates quality with LLMJudge across 4 dimensions
      4. Returns structured ComboResult objects

    Retrievers must already be set up (setup_and_time called) before passing
    them in — this class only measures query-time performance.
    """

    def __init__(
        self,
        retrievers: dict[str, BaseRetriever],
        generators: dict[str, BaseGenerator],
        questions: list[str],
        ground_truth: dict[str, str],
        top_k: int = 5,
    ):
        self.retrievers = retrievers
        self.generators = generators
        self.questions = questions
        self.ground_truth = ground_truth
        self.top_k = top_k
        self.judge = LLMJudge()
        self.results: list[ComboResult] = []

    def run_all(self) -> list[ComboResult]:
        """
        Run every retriever × generator pair with interleaved question order.

        Each question is sent to all combos before moving to the next question,
        so every retriever faces the same LLM API conditions (rate limits, load)
        and generation latency is comparable across combos.

        Order:
            Q1 → [Vector+LLaMA, BM25+LLaMA, Hybrid+LLaMA]
            Q2 → [Vector+LLaMA, BM25+LLaMA, Hybrid+LLaMA]
            ...
        """
        combos = [
            (ret_name, retriever, gen_name, generator)
            for ret_name, retriever in self.retrievers.items()
            for gen_name, generator in self.generators.items()
        ]

        # Accumulators keyed by (retriever_name, generator_name)
        acc: dict[tuple, dict] = {
            (r, g): {"retrieval_times": [], "generation_times": [], "token_counts": [], "samples": [], "per_question": []}
            for r, _, g, _ in combos
        }

        # Outer loop: questions — inner loop: combos
        for q_i, q in enumerate(self.questions):
            print(f"\nQ{q_i + 1}/{len(self.questions)}: {q[:65]}...")
            for ret_name, retriever, gen_name, generator in combos:
                ret_result = retriever.retrieve_and_time(q, top_k=self.top_k)
                gen_result = generator.generate_and_time(q, ret_result.chunks)

                key = (ret_name, gen_name)
                acc[key]["retrieval_times"].append(ret_result.latency_ms)
                acc[key]["generation_times"].append(gen_result.latency_ms)
                acc[key]["token_counts"].append(gen_result.tokens_used)
                acc[key]["samples"].append({
                    "question": q,
                    "answer": gen_result.answer,
                    "contexts": ret_result.chunks,
                    "ground_truth": self.ground_truth.get(q, ""),
                })
                acc[key]["per_question"].append({
                    "question": q,
                    "retrieval_ms": round(ret_result.latency_ms, 1),
                    "generation_ms": round(gen_result.latency_ms, 1),
                    "tokens": gen_result.tokens_used,
                })
                print(
                    f"  {ret_name}+{gen_name}: "
                    f"ret={ret_result.latency_ms:.0f}ms  gen={gen_result.latency_ms:.0f}ms"
                )

        # Judge all combos after all questions are collected
        self.results = []
        n = len(self.questions)
        for ret_name, retriever, gen_name, generator in combos:
            key = (ret_name, gen_name)
            data = acc[key]
            print(f"\nJudging {ret_name}+{gen_name} ({n} samples)...")
            avg_score, _ = self.judge.evaluate_batch(data["samples"])

            rt = data["retrieval_times"]
            gt = data["generation_times"]
            result = ComboResult(
                retriever_name=ret_name,
                generator_name=gen_name,
                setup_metrics=retriever.setup_metrics,
                avg_retrieval_ms=sum(rt) / n,
                avg_generation_ms=sum(gt) / n,
                avg_total_ms=(sum(rt) + sum(gt)) / n,
                avg_tokens=sum(data["token_counts"]) / n,
                judge_scores=avg_score.to_dict(),
                per_question=data["per_question"],
            )
            self.results.append(result)
            print(
                f"  avg score: {result.judge_scores.get('average', 0):.3f} | "
                f"avg latency: {result.avg_total_ms:.0f}ms"
            )

        return self.results
