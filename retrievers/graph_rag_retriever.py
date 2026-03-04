# retrievers/graph_rag_retriever.py
import json
import os
import time
from collections import deque

import networkx as nx
from groq import Groq
from dotenv import load_dotenv

from retrievers.base import BaseRetriever, RetrievalResult
from utils.profiler import measure_storage

load_dotenv()


class GraphRAGRetriever(BaseRetriever):
    """
    Graph RAG retriever: builds a knowledge graph from document chunks using
    LLM-extracted triples, then traverses the graph to find relevant chunks.

    Setup (one-time, expensive — uses LLM API calls per chunk):
        1. For each chunk, call Groq to extract (subject, relation, object) triples.
        2. Build a NetworkX MultiDiGraph:
              entity nodes  : subjects and objects from triples (lowercased)
              chunk nodes   : one node per document chunk
              entity→entity : the extracted relation
              entity→chunk  : "MENTIONED_IN" (which chunk the triple came from)
        3. Save graph to disk as JSON (measures storage_mb).

    If graph_path already exists, the graph is loaded from disk — no LLM calls.
    Delete the file to force a fresh rebuild.

    Retrieval (fast — pure graph traversal, no LLM call):
        1. Seed: find entity nodes whose name appears in the query text.
        2. BFS expand from seed entities up to max_hops.
        3. Collect chunk nodes reachable from the expanded entity set.
        4. Fallback if empty: broaden to entities sharing any query word (len > 3).

    Advantage over dense/sparse: multi-hop reasoning — can surface chunks
    connected through intermediate concepts not present in the query.
    """

    _TRIPLE_PROMPT = """\
Extract factual (subject, relation, object) triples from the text below.

Rules:
- Use concise noun phrases for subject and object.
- Use short active-voice verb phrases for relation.
- Extract 3–8 triples. Return [] if there is nothing factual.
- Return ONLY a valid JSON array, no markdown, no explanation.

Format: [{"subject": "...", "relation": "...", "object": "..."}, ...]

TEXT:
{text}

JSON:"""

    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",  # fast model for extraction
        max_hops: int = 2,
        graph_path: str = "data/knowledge_graph.json",
    ):
        super().__init__(name="GraphRAGRetriever")
        self.model_name = model_name
        self.max_hops = max_hops
        self.graph_path = graph_path
        self._graph = nx.MultiDiGraph()
        self._chunk_lookup: dict[str, str] = {}  # chunk_id → chunk text
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # ── Triple extraction (LLM) ───────────────────────────────────────

    def _extract_triples(self, text: str, retries: int = 3) -> list[dict]:
        """
        Call Groq to extract triples from a chunk of text.
        Returns list of {"subject": ..., "relation": ..., "object": ...}.
        Retries up to `retries` times on rate-limit (429) errors.
        """
        for attempt in range(retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": self._TRIPLE_PROMPT.format(text=text[:2000]),
                        }
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                raw = response.choices[0].message.content.strip()
                # Strip markdown code fences if present
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                triples = json.loads(raw)
                return [t for t in triples if t.get("subject") and t.get("object")]
            except Exception as e:
                if "429" in str(e) and attempt < retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    print(f"[{self.name}] Rate limited — retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    return []
        return []

    # ── Graph building ────────────────────────────────────────────────

    def _build_graph(self, chunks: list) -> int:
        """Extract triples from all chunks and populate self._graph. Returns triple count."""
        total_triples = 0
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            self._chunk_lookup[chunk_id] = chunk.text
            self._graph.add_node(chunk_id, label="chunk")

            triples = self._extract_triples(chunk.text)
            for triple in triples:
                subj = str(triple.get("subject", "")).strip().lower()
                rel = str(triple.get("relation", "")).strip()
                obj = str(triple.get("object", "")).strip().lower()
                if not subj or not obj:
                    continue

                # Entity nodes
                self._graph.add_node(subj, label="entity")
                self._graph.add_node(obj, label="entity")

                # Entity → Entity edge (the semantic relation)
                self._graph.add_edge(subj, obj, relation=rel, chunk_id=chunk_id)

                # Entity ↔ Chunk edges (link triples back to source)
                self._graph.add_edge(subj, chunk_id, relation="MENTIONED_IN")
                self._graph.add_edge(chunk_id, obj, relation="MENTIONS")

                total_triples += 1

            if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
                print(
                    f"[{self.name}] {i + 1}/{len(chunks)} chunks — "
                    f"{total_triples} triples extracted"
                )

        return total_triples

    # ── BaseRetriever interface ────────────────────────────────────────

    def setup(self, chunks: list) -> None:
        """
        Build the knowledge graph.

        If graph_path already exists on disk, loads it (skips LLM extraction).
        Delete the file to force a full rebuild.
        """
        self._chunk_lookup = {}
        self._graph = nx.MultiDiGraph()

        if os.path.exists(self.graph_path):
            print(f"[{self.name}] Loading cached graph from {self.graph_path} ...")
            t0 = time.perf_counter()
            with open(self.graph_path) as f:
                self._graph = nx.node_link_graph(json.load(f))
            # Rebuild chunk lookup from graph node attributes
            for i, chunk in enumerate(chunks):
                self._chunk_lookup[f"chunk_{i}"] = chunk.text
            self.setup_metrics.indexing_ms = (time.perf_counter() - t0) * 1000
            self.setup_metrics.storage_mb = measure_storage([self.graph_path])
            print(
                f"[{self.name}] Loaded — "
                f"{self._graph.number_of_nodes()} nodes | "
                f"{self._graph.number_of_edges()} edges"
            )
            return

        # Fresh build: extract triples via LLM
        print(f"[{self.name}] Extracting triples with {self.model_name} ...")
        t0 = time.perf_counter()
        total_triples = self._build_graph(chunks)
        self.setup_metrics.indexing_ms = (time.perf_counter() - t0) * 1000

        # Persist graph to disk
        os.makedirs(os.path.dirname(self.graph_path) or ".", exist_ok=True)
        with open(self.graph_path, "w") as f:
            json.dump(nx.node_link_data(self._graph), f)
        self.setup_metrics.storage_mb = measure_storage([self.graph_path])

        print(
            f"[{self.name}] Graph saved → {self.graph_path} | "
            f"{self._graph.number_of_nodes()} nodes | "
            f"{self._graph.number_of_edges()} edges | "
            f"{total_triples} triples"
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """
        Traverse the knowledge graph to find chunks relevant to query.

        Step 1 — Seed: find entity nodes whose name is a substring of the query.
        Step 2 — BFS expand up to max_hops through entity→entity edges.
        Step 3 — Collect chunk nodes reachable from the expanded entity set.
        Fallback — if no chunks found, broaden seed to entities sharing any
                   query word (length > 3).
        """
        seeds = self._seed_entities(query)
        chunk_ids = self._bfs_expand(seeds) if seeds else set()

        # Fallback: broaden to entities sharing individual query words
        if not chunk_ids:
            words = {w for w in query.lower().split() if len(w) > 3}
            broad_seeds = {
                n
                for n, d in self._graph.nodes(data=True)
                if d.get("label") == "entity" and any(w in n for w in words)
            }
            chunk_ids = self._bfs_expand(broad_seeds) if broad_seeds else set()

        results = [
            self._chunk_lookup[cid]
            for cid in chunk_ids
            if cid in self._chunk_lookup
        ]

        # Last resort: return highest-degree chunk nodes
        if not results:
            chunk_nodes = sorted(
                [
                    (n, self._graph.degree(n))
                    for n, d in self._graph.nodes(data=True)
                    if d.get("label") == "chunk" and n in self._chunk_lookup
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            results = [self._chunk_lookup[n] for n, _ in chunk_nodes]

        return results[:top_k]

    # ── Graph helpers ─────────────────────────────────────────────────

    def _seed_entities(self, query: str) -> set[str]:
        """Return entity node names that appear as substrings in the query."""
        q_lower = query.lower()
        return {
            n
            for n, d in self._graph.nodes(data=True)
            if d.get("label") == "entity" and n in q_lower
        }

    def _bfs_expand(self, seeds: set[str]) -> set[str]:
        """BFS from seed entity nodes; collect reachable chunk node IDs."""
        visited: set[str] = set()
        chunk_ids: set[str] = set()
        queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds)

        while queue:
            node, depth = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            if self._graph.nodes[node].get("label") == "chunk":
                chunk_ids.add(node)
                continue  # don't expand further from chunk nodes

            if depth >= self.max_hops:
                continue

            for _, neighbor in self._graph.out_edges(node):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
            for neighbor, _ in self._graph.in_edges(node):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

        return chunk_ids

    # ── Graph inspection ──────────────────────────────────────────────

    def top_entities(self, n: int = 10) -> list[tuple[str, int]]:
        """Return the top-n most connected entity nodes (by total degree)."""
        return sorted(
            [
                (node, self._graph.degree(node))
                for node, data in self._graph.nodes(data=True)
                if data.get("label") == "entity"
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:n]
