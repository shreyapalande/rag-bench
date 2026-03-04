# RAG Retriever Benchmark

A benchmarking framework for comparing Retrieval-Augmented Generation (RAG) strategies. It measures retriever quality, latency, setup cost, and memory usage across multiple retrieval methods paired with LLM generators.

## What It Does

Given a set of PDF documents and benchmark questions, this project:

1. **Chunks** PDFs into overlapping text segments
2. **Indexes** them with six different retrieval strategies
3. **Runs** each question through every retriever → generator combination
4. **Evaluates** answer quality with an LLM judge (4 dimensions)
5. **Reports** quality scores, latency, setup cost, and cache performance in JSON, CSV, and Markdown

---

## Retrievers

| Retriever    | Method                                | Key Dependency                       | Strength                          |
| ------------ | ------------------------------------- | ------------------------------------ | --------------------------------- |
| **Vector**   | Dense — FAISS + sentence-transformers | `faiss-cpu`, `sentence-transformers` | Semantic similarity               |
| **BM25**     | Sparse — TF-IDF keyword ranking       | `rank-bm25`                          | Exact keyword matches             |
| **Hybrid**   | RRF fusion of Vector + BM25           | Both above                           | Balanced coverage                 |
| **KVCache**  | Exact-match Redis cache over Vector   | `redis`                              | Zero-latency repeat queries       |
| **SemCache** | Semantic Redis cache over Vector      | `redis`                              | Cache hits on paraphrased queries |
| **GraphRAG** | Knowledge graph traversal (NetworkX)  | `networkx`, Groq API                 | Multi-hop entity reasoning        |

### How Each Retriever Works

**Vector** — Embeds all chunks with `all-MiniLM-L6-v2` at setup, builds a FAISS `IndexFlatIP` (cosine similarity). At query time, embeds the query and returns the top-k nearest chunks.

**BM25** — Tokenizes all chunks at setup using `rank-bm25`. At query time, scores chunks by BM25 Okapi and returns the top-k.

**Hybrid** — Sets up both Vector and BM25 sub-retrievers, then fuses their ranked results using Reciprocal Rank Fusion (`score = 1 / (k + rank)`, default `k=60`).

**KVCache** — Wraps Vector with Redis exact-match caching. Cache key = `sha256(normalized_query)[:16]`. On a hit, returns cached chunks in ~1ms. On a miss, runs FAISS search and stores the result. Supports `warm_cache()` to pre-populate before benchmarking.

**SemCache** — Wraps Vector with Redis semantic caching. On each query: embeds the query, scans all `sem_cache::*` Redis keys, computes cosine similarity against each cached embedding. If the best match ≥ threshold (default 0.85), returns cached chunks. Falls back to FAISS on a miss. Catches paraphrased queries that KVCache misses.

**GraphRAG** — At setup, calls Groq LLM (llama-3.1-8b-instant) to extract `(subject, relation, object)` triples from each chunk. Builds a NetworkX MultiDiGraph with entity nodes, chunk nodes, and edges. Graph is saved to `data/knowledge_graph.json` and reloaded on subsequent runs. At query time, finds entity nodes matching the query text, BFS-expands up to `max_hops=2`, and returns chunks reachable from the expanded entity set.

---

## Generators

| Generator             | Model                     | API       |
| --------------------- | ------------------------- | --------- |
| **LLaMA** (default)   | `llama-3.3-70b-versatile` | Groq      |
| **Gemini** (optional) | `gemini-2.0-flash`        | Google AI |

---

## Evaluation

Answers are judged by an LLM judge (Groq LLaMA) on four dimensions:

| Metric                | What it measures                                                        | Reference-free? |
| --------------------- | ----------------------------------------------------------------------- | --------------- |
| **Faithfulness**      | Does the answer only use information from the retrieved context?        | Yes             |
| **Answer Relevancy**  | Does the answer actually address the question?                          | Yes             |
| **Context Relevancy** | Does the retrieved context contain information useful for the question? | Yes             |
| **Completeness**      | Does the answer cover all aspects of the reference ground truth?        | No (needs GT)   |

All scores are 0.0–1.0. The `average` is the mean of all four.

> **Note:** The ground truth in `data/ground_truth.json` is LLM-generated, making `completeness` circular. The first three metrics are reference-free and valid for comparing retrieval strategies.

---

## Project Structure

```
rag project/
├── config.py                    # Dataclass-based config (EmbeddingConfig, DataConfig, etc.)
├── test_eval.py                 # Main benchmark entry point
│
├── retrievers/
│   ├── base.py                  # BaseRetriever + RetrievalResult
│   ├── vector_retriever.py      # FAISS dense retrieval
│   ├── bm25_retriever.py        # BM25 sparse retrieval
│   ├── hybrid_retriever.py      # RRF fusion of Vector + BM25
│   ├── kv_cache_retriever.py    # Exact-match Redis cache over Vector
│   ├── semantic_cache_retriever.py  # Semantic Redis cache over Vector
│   └── graph_rag_retriever.py   # Knowledge graph traversal (NetworkX)
│
├── generators/
│   ├── base.py                  # BaseGenerator + GenerationResult
│   ├── groq_generator.py        # Groq LLaMA
│   └── gemini_generator.py      # Google Gemini
│
├── benchmark/
│   ├── runner.py                # BenchmarkRunner — orchestrates retriever × generator pairs
│   └── reporter.py              # Saves JSON / CSV / Markdown reports + prints console table
│
├── evaluation/
│   ├── llm_judge.py             # LLM-as-judge scoring (4 dimensions)
│   └── ragas_eval.py            # RAGAS evaluation (alternative)
│
├── utils/
│   ├── chunker.py               # PDF loading + character-based chunking
│   └── profiler.py              # SetupMetrics, MemoryTracker (psutil RSS), QueryMetrics
│
├── tests/
│   ├── test_kv_cache.py         # Standalone KVCache test (cold/warm/latency)
│   ├── test_semantic_cache.py   # Standalone SemCache test (paraphrase test included)
│   └── test_graph_rag.py        # Standalone GraphRAG test (graph stats + retrieval traces)
│
├── data/
│   ├── doc1.pdf … doc4.pdf      # Source PDF documents
│   ├── processed_chunks.json    # Pre-chunked text (800 chars, 150 overlap)
│   ├── ground_truth.json        # 5 benchmark Q&A pairs
│   ├── faiss_index.bin          # Pre-built FAISS index (optional)
│   └── knowledge_graph.json     # Cached GraphRAG knowledge graph (generated on first run)
│
└── reports/                     # Timestamped benchmark outputs
    └── benchmark_results.json
```

---

## Setup

### Prerequisites

- Python 3.10+
- Redis (for KVCache and SemCache retrievers)
- A Groq API key (for LLaMA generator and GraphRAG triple extraction)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start Redis

```bash
# Docker (recommended)
docker run -p 6379:6379 redis

# Or if Redis is installed locally
redis-server
```

### Configure API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here   # optional
REDIS_URL=redis://localhost:6379/0        # optional, this is the default
```

---

## Running

### Full benchmark (all retrievers × all generators)

```bash
python test_eval.py
```

This will:

1. Load chunks from `data/processed_chunks.json`
2. Set up all 6 retrievers (Vector embeds all chunks, GraphRAG extracts triples via Groq)
3. Warm KVCache and SemCache with the first half of benchmark questions
4. Run all questions through every retriever × generator combination
5. Judge all answers with LLMJudge
6. Save reports to `reports/`

> **GraphRAG first run:** Triple extraction via Groq takes time proportional to the number of chunks. The graph is saved to `data/knowledge_graph.json` and reused on subsequent runs. Delete this file to force a rebuild.

---

## Reports

Each benchmark run generates the following file in `reports/`:

**`benchmark_results.json`** — Full structured data including per-question latency and cache hits.

```
## Quality Rankings         (sorted by avg score)
## Query Latency            (avg retrieval + generation ms)
## Setup Cost               (per retriever: total, embed, tokenize, index, memory)
## Cache Performance        (hit rate, hits, misses — KVCache and SemCache only)
```

A summary table is also printed to the console at the end of each run.

---

## Setup Metrics Explained

Each retriever records phase-level timing during `setup_and_time()`:

| Metric           | Recorded by                                                                   |
| ---------------- | ----------------------------------------------------------------------------- |
| `embedding_ms`   | Vector, KVCache, SemCache — time to encode all chunks                         |
| `tokenizing_ms`  | BM25, Hybrid — time to tokenize all chunks                                    |
| `indexing_ms`    | All — time to build the index structure (FAISS / BM25Okapi / knowledge graph) |
| `memory_peak_mb` | All — RSS process memory growth during setup (via `psutil`)                   |
| `storage_mb`     | GraphRAG — size of the saved `knowledge_graph.json`                           |

Memory is measured using `psutil` RSS delta (not `tracemalloc`) to capture C-extension allocations from numpy, FAISS, and PyTorch.
