# benchmark/reporter.py
import csv
import json
import os
from datetime import datetime

from benchmark.runner import ComboResult


class BenchmarkReporter:
    """Generates JSON, CSV, and Markdown reports from benchmark results."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_all(self, results: list[ComboResult], prefix: str = "benchmark") -> str:
        """Save JSON + CSV + Markdown and print the summary table."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{self.output_dir}/{prefix}_{ts}"

        self.save_json(results, f"{base}.json")
        self.save_csv(results, f"{base}.csv")
        md_path = self.save_markdown(results, f"{base}.md")
        self.print_table(results)

        print(f"\nReports saved → {self.output_dir}/")
        return md_path

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
                "per_question": r.per_question,
            }
            for r in results
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ── CSV ───────────────────────────────────────────────────────────

    def save_csv(self, results: list[ComboResult], path: str):
        rows = [r.summary_row() for r in results]
        if not rows:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # ── Markdown ──────────────────────────────────────────────────────

    def save_markdown(self, results: list[ComboResult], path: str) -> str:
        by_quality = sorted(results, key=lambda r: r.judge_scores.get("average", 0), reverse=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines = [
            "# RAG Benchmark Report",
            f"_Generated: {ts}_\n",

            "## Quality Rankings\n",
            f"| {'Combo':<25} | {'Faith':>6} | {'Relev':>6} | {'CtxRel':>7} | {'Compl':>6} | {'Avg':>6} |",
            f"|{'-'*27}|{'-'*8}|{'-'*8}|{'-'*9}|{'-'*8}|{'-'*8}|",
        ]
        for r in by_quality:
            s = r.judge_scores
            lines.append(
                f"| {r.combo:<25} "
                f"| {s.get('faithfulness', 0):>6.3f} "
                f"| {s.get('answer_relevancy', 0):>6.3f} "
                f"| {s.get('context_relevancy', 0):>7.3f} "
                f"| {s.get('completeness', 0):>6.3f} "
                f"| {s.get('average', 0):>6.3f} |"
            )

        lines += [
            "\n## Query Latency\n",
            f"| {'Combo':<25} | {'Retrieve':>9} | {'Generate':>10} | {'Total':>8} | {'Tokens':>7} |",
            f"|{'-'*27}|{'-'*11}|{'-'*12}|{'-'*10}|{'-'*9}|",
        ]
        for r in by_quality:
            lines.append(
                f"| {r.combo:<25} "
                f"| {r.avg_retrieval_ms:>8.1f}ms "
                f"| {r.avg_generation_ms:>9.1f}ms "
                f"| {r.avg_total_ms:>7.1f}ms "
                f"| {r.avg_tokens:>7.0f} |"
            )

        lines += [
            "\n## Setup Cost (per retriever)\n",
            f"| {'Retriever':<20} | {'Total':>8} | {'Embed':>8} | {'Tokenize':>9} | {'Index':>8} | {'Mem (MB)':>9} |",
            f"|{'-'*22}|{'-'*10}|{'-'*10}|{'-'*11}|{'-'*10}|{'-'*11}|",
        ]
        seen: set[str] = set()
        for r in by_quality:
            if r.retriever_name not in seen:
                seen.add(r.retriever_name)
                m = r.setup_metrics
                lines.append(
                    f"| {r.retriever_name:<20} "
                    f"| {m.total_ms:>7.0f}ms "
                    f"| {m.embedding_ms:>7.0f}ms "
                    f"| {m.tokenizing_ms:>8.0f}ms "
                    f"| {m.indexing_ms:>7.0f}ms "
                    f"| {m.memory_peak_mb:>9.2f} |"
                )

        content = "\n".join(lines) + "\n"
        with open(path, "w") as f:
            f.write(content)
        return path

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
