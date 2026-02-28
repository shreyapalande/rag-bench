# config.py
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    dimension: int = 384


@dataclass
class DataConfig:
    data_dir: str = "data"
    chunks_path: str = "data/processed_chunks.json"
    faiss_index_path: str = "data/faiss_index.bin"
    chunks_pkl_path: str = "data/chunks.pkl"
    ground_truth_path: str = "data/ground_truth.json"


@dataclass
class BenchmarkConfig:
    top_k: int = 5
    rrf_k: int = 60
    output_dir: str = "reports"


@dataclass
class APIConfig:
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-2.0-flash"


@dataclass
class AppConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    api: APIConfig = field(default_factory=APIConfig)


CONFIG = AppConfig()
