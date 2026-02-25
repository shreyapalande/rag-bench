import os
import json
import fitz  # PyMuPDF

# -------- CONFIG --------
DATA_DIR = "../data"        # Path to your PDFs
CHUNK_SIZE = 800            # Number of characters per chunk
CHUNK_OVERLAP = 150         # Overlap between chunks
OUTPUT_FILE = "../data/processed_chunks.json"

# -------- HELPER FUNCTIONS --------
def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# -------- MAIN PIPELINE --------
def load_and_chunk_pdfs(data_dir=DATA_DIR):
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    all_chunks = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        print(f"Processing {pdf_file}...")
        text = extract_text_from_pdf(pdf_path)
        text_chunks = chunk_text(text)

        for idx, chunk in enumerate(text_chunks):
            all_chunks.append({
                "chunk_id": f"{os.path.splitext(pdf_file)[0]}_{idx}",
                "file_name": pdf_file,
                "text": chunk,
                "chunk_index": idx
            })

    # Save chunks for reuse
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(pdf_files)} PDFs into {len(all_chunks)} chunks.")
    return all_chunks

# -------- ENTRY POINT --------
if __name__ == "__main__":
    chunks = load_and_chunk_pdfs()
    print("Sample chunk:")
    print(chunks[0]["text"][:500])