# generators/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time

@dataclass
class GenerationResult:
    answer: str
    latency_ms: float
    tokens_used: int = 0
    metadata: dict = field(default_factory=dict)

class BaseGenerator(ABC):
    def __init__(self, name: str):
        self.name = name

    def generate_and_time(self, query: str, contexts: list[str]) -> GenerationResult:
        start = time.perf_counter()
        result = self.generate(query, contexts)
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    @abstractmethod
    def generate(self, query: str, contexts: list[str]) -> GenerationResult:
        pass