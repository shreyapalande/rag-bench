from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -------- CONFIG --------
CHUNKS_FILE = "../data/processed_chunks.json"  # your chunks JSON
FAISS_INDEX_FILE = "../data/faiss_index.bin"   # your FAISS index
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0"
TOP_K = 5

# -------- LOAD CHUNKS AND FAISS INDEX --------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

index = faiss.read_index(FAISS_INDEX_FILE)

# -------- LOAD EMBEDDING MODEL --------
embed_model = SentenceTransformer(EMBEDDING_MODEL)

# -------- LOAD LLM --------
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, device_map="auto")
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,  # adjust if needed
)

# -------- FUNCTIONS --------
def retrieve_chunks(question: str, top_k: int = TOP_K):
    q_embedding = embed_model.encode([question]).astype("float32")
    D, I = index.search(q_embedding, top_k)
    retrieved = [chunks[i]["text"] for i in I[0]]
    return retrieved

def generate_answer(question: str):
    retrieved_chunks = retrieve_chunks(question)
    context = "\n".join(retrieved_chunks)
    prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    response = generator(prompt, do_sample=True, temperature=0.7)[0]['generated_text']
    return response

# -------- EXAMPLE --------
if __name__ == "__main__":
    question = "What is latency-aware pruning in embedded CNNs?"
    answer = generate_answer(question)
    print("Answer:", answer)