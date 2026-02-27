# utils/chunker.py
import os
import json
import fitz
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: str
    file_name: str
    text: str
    chunk_index: int

class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF."""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping character-based chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def process_directory(self, data_dir: str) -> list[Chunk]:
        """Load all PDFs from directory and return chunks."""
        pdf_files = list(Path(data_dir).glob("**/*.pdf"))
        all_chunks = []

        for pdf_path in pdf_files:
            print(f"Processing {pdf_path.name}...")
            text = self.extract_text(str(pdf_path))
            text_chunks = self.chunk_text(text)

            for idx, chunk_text in enumerate(text_chunks):
                all_chunks.append(Chunk(
                    chunk_id=f"{pdf_path.stem}_{idx}",
                    file_name=pdf_path.name,
                    text=chunk_text,
                    chunk_index=idx
                ))

        print(f"Processed {len(pdf_files)} PDFs → {len(all_chunks)} chunks")
        return all_chunks

    def save_chunks(self, chunks: list[Chunk], output_path: str):
        """Save chunks to JSON for reuse."""
        data = [
            {
                "chunk_id": c.chunk_id,
                "file_name": c.file_name,
                "text": c.text,
                "chunk_index": c.chunk_index
            }
            for c in chunks
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(chunks)} chunks to {output_path}")

    def load_chunks(self, input_path: str) -> list[Chunk]:
        """Load chunks from JSON."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Chunk(**item) for item in data]