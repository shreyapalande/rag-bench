import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------- CONFIG --------
CHUNKS_FILE = "../data/processed_chunks.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast
VECTOR_DIM = 384  # depends on model
FAISS_INDEX_FILE = "../data/faiss_index.bin"

# -------- LOAD CHUNKS --------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]

# -------- GENERATE EMBEDDINGS --------
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# -------- BUILD FAISS INDEX --------
index = faiss.IndexFlatL2(VECTOR_DIM)  # L2 distance
index.add(embeddings)
faiss.write_index(index, FAISS_INDEX_FILE)

print(f"Stored {len(chunks)} chunk embeddings in FAISS index.")