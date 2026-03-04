# benchmark/reporter.py
import json
import os

from benchmark.runner import ComboResult


class BenchmarkReporter:
    """Saves benchmark results as JSON and prints a summary table."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_all(self, results: list[ComboResult], filename: str = "benchmark_results.json") -> str:
        """Save JSON and print the summary table."""
        path = f"{self.output_dir}/{filename}"
        self.save_json(results, path)
        self.print_table(results)
        print(f"\nResults saved → {path}")
        return path

    # ── JSON ──────────────────────────────────────────────────────────

    def save_json(self, results: list[ComboResult], path: str):
        data = [
            {
                "combo": r.combo,
                "setup_metrics": r.setup_metrics.to_dict(),
                "latency": {
                    "avg_retrieval_ms": round(r.avg_retrieval_ms, 1),
                    "avg_generation_ms": round(r.avg_generation_ms, 1),
                    "avg_total_ms": round(r.avg_total_ms, 1),
                    "avg_tokens": round(r.avg_tokens, 1),
                },
                "quality": r.judge_scores,
                "cache_stats": r.cache_stats or None,
                "per_question": r.per_question,
            }
            for r in results
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Console table ─────────────────────────────────────────────────

    def print_table(self, results: list[ComboResult]):
        by_quality = sorted(results, key=lambda r: r.judge_scores.get("average", 0), reverse=True)

        W = 108
        print("\n" + "=" * W)
        print("BENCHMARK RESULTS — Quality (sorted by avg score)")
        print("=" * W)
        print(
            f"{'Combo':<28} {'Faith':>7} {'Relev':>7} {'CtxRel':>8} {'Compl':>7} {'Avg':>6}"
            f"  │  {'Retrieve':>9} {'Generate':>10} {'Total':>8}"
        )
        print("─" * W)
        for r in by_quality:
            s = r.judge_scores
            print(
                f"{r.combo:<28} "
                f"{s.get('faithfulness', 0):>7.3f} "
                f"{s.get('answer_relevancy', 0):>7.3f} "
                f"{s.get('context_relevancy', 0):>8.3f} "
                f"{s.get('completeness', 0):>7.3f} "
                f"{s.get('average', 0):>6.3f}"
                f"  │  "
                f"{r.avg_retrieval_ms:>7.0f}ms "
                f"{r.avg_generation_ms:>9.0f}ms "
                f"{r.avg_total_ms:>7.0f}ms"
            )

        print("\n" + "─" * 65)
        print("SETUP COST (per retriever)")
        print("─" * 65)
        print(f"{'Retriever':<20} {'Total':>9} {'Embed':>9} {'Tokenize':>10} {'Index':>9} {'Mem':>8}")
        print("─" * 65)
        seen: set[str] = set()
        for r in by_quality:
            if r.retriever_name not in seen:
                seen.add(r.retriever_name)
                m = r.setup_metrics
                print(
                    f"{r.retriever_name:<20} "
                    f"{m.total_ms:>7.0f}ms "
                    f"{m.embedding_ms:>7.0f}ms "
                    f"{m.tokenizing_ms:>8.0f}ms "
                    f"{m.indexing_ms:>7.0f}ms "
                    f"{m.memory_peak_mb:>6.1f}MB"
                )
        print("─" * 65)

        cache_results = [r for r in by_quality if r.cache_stats]
        if cache_results:
            print("\n" + "─" * 45)
            print("CACHE PERFORMANCE")
            print("─" * 45)
            print(f"{'Combo':<28} {'Hit Rate':>9} {'Hits':>6} {'Misses':>7}")
            print("─" * 45)
            for r in cache_results:
                c = r.cache_stats
                print(
                    f"{r.combo:<28} "
                    f"{c['hit_rate']:>9.1%} "
                    f"{c['hits']:>6} "
                    f"{c['misses']:>7}"
                )
            print("─" * 45)
