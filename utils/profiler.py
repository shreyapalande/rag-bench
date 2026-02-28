# utils/profiler.py
import os
from dataclasses import dataclass

import psutil


@dataclass
class SetupMetrics:
    """Resource usage and per-phase timing captured during retriever setup."""
    total_ms: float = 0.0
    embedding_ms: float = 0.0    # Time to encode chunks with a transformer model
    tokenizing_ms: float = 0.0   # Time to tokenize chunks for BM25
    indexing_ms: float = 0.0     # Time to build the index structure (FAISS / BM25Okapi)
    memory_peak_mb: float = 0.0  # Peak RSS process memory growth during setup (MB)
    storage_mb: float = 0.0      # Disk size of persisted index files (if saved)

    def to_dict(self) -> dict:
        return {
            "total_ms": round(self.total_ms, 1),
            "embedding_ms": round(self.embedding_ms, 1),
            "tokenizing_ms": round(self.tokenizing_ms, 1),
            "indexing_ms": round(self.indexing_ms, 1),
            "memory_peak_mb": round(self.memory_peak_mb, 2),
            "storage_mb": round(self.storage_mb, 2),
        }


@dataclass
class QueryMetrics:
    """Per-query timing and token usage."""
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    tokens_used: int = 0

    def to_dict(self) -> dict:
        return {
            "retrieval_ms": round(self.retrieval_ms, 1),
            "generation_ms": round(self.generation_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "tokens_used": self.tokens_used,
        }


class MemoryTracker:
    """
    Context manager that measures peak RSS process memory growth during setup.

    Captures the OS-level resident set size (RSS) before and after the block,
    including memory allocated by C extensions (numpy, FAISS, torch) that
    tracemalloc cannot see.

    Usage:
        with MemoryTracker() as mem:
            do_work()
        print(mem.peak_mb)   # MB of RSS added during the block
    """

    def __init__(self):
        self.peak_mb: float = 0.0
        self._process = psutil.Process()

    def __enter__(self):
        self._rss_before = self._process.memory_info().rss
        return self

    def __exit__(self, *args):
        rss_after = self._process.memory_info().rss
        growth = max(rss_after - self._rss_before, 0)
        self.peak_mb = growth / 1e6


def measure_storage(paths: list[str]) -> float:
    """Return total size in MB of the given file paths that exist on disk."""
    return sum(
        os.path.getsize(p) for p in paths if os.path.exists(p)
    ) / 1e6
