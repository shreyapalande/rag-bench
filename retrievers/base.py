# retrievers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import time

@dataclass
class RetrievalResult:
    chunks: list[str]           # Retrieved text chunks
    latency_ms: float           # Query latency in milliseconds
    metadata: dict = field(default_factory=dict)

class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.
    Every retriever must implement setup() and retrieve().
    """

    def __init__(self, name: str):
        self.name = name
        self.setup_time_ms: float = 0.0
        self._is_setup: bool = False

    def setup_and_time(self, chunks: list) -> float:
        """Setup retriever and record setup time."""
        start = time.perf_counter()
        self.setup(chunks)
        self.setup_time_ms = (time.perf_counter() - start) * 1000
        self._is_setup = True
        print(f"[{self.name}] Setup complete in {self.setup_time_ms:.1f}ms")
        return self.setup_time_ms

    def retrieve_and_time(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve chunks and record latency."""
        if not self._is_setup:
            raise RuntimeError(f"[{self.name}] Call setup() before retrieve()")
        start = time.perf_counter()
        chunks = self.retrieve(query, top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            chunks=chunks,
            latency_ms=latency_ms,
            metadata={"retriever": self.name, "top_k": top_k}
        )

    @abstractmethod
    def setup(self, chunks: list) -> None:
        """Index chunks. Called once before any queries."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Return top_k relevant text chunks for query."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"