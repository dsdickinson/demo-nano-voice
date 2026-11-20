"""
Document ingestion and text chunking pipeline for local RAG.
Reads markdown and text files, cleans whitespace, and splits into overlapping chunks with metadata.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source_file: str
    char_start: int
    char_end: int


class DocumentIngester:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source_file: str = "") -> List[DocumentChunk]:
        """Split a string into overlapping character chunks preserving word boundaries where possible."""
        text = text.strip()
        if not text:
            return []

        chunks: List[DocumentChunk] = []
        start = 0
        text_len = len(text)
        chunk_idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # If not at the end of the text, try to break at last whitespace
            if end < text_len:
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append(DocumentChunk(
                    chunk_id=f"{os.path.basename(source_file)}_{chunk_idx}",
                    text=chunk_content,
                    source_file=source_file,
                    char_start=start,
                    char_end=end
                ))
                chunk_idx += 1

            if end >= text_len:
                break
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def ingest_directory(self, directory_path: str) -> List[DocumentChunk]:
        """Recursively scan directory for .md and .txt files and chunk all found documents."""
        all_chunks: List[DocumentChunk] = []
        if not os.path.exists(directory_path):
            return all_chunks

        for root, _, files in os.walk(directory_path):
            for file in sorted(files):
                if file.endswith((".md", ".txt")):
                    full_path = os.path.join(root, file)
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    chunks = self.chunk_text(content, source_file=full_path)
                    all_chunks.extend(chunks)

        return all_chunks
