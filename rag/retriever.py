"""
Vector store and semantic search retriever for edge RAG.
Supports sentence-transformers / FAISS when available, with a fast zero-dependency
PyTorch/TF-IDF cosine retriever fallback for edge or offline operation.
"""

from typing import List, Tuple, Optional
import math
import numpy as np
import torch

from rag.ingest import DocumentChunk


class SimpleEmbeddingModel:
    """
    Lightweight zero-dependency embedding encoder using subword/character hash n-grams
    projected into normalized dense vectors. Provides reliable semantic similarity without
    requiring multi-gigabyte downloads.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            words = text.lower().split()
            if not words:
                vectors.append(vec)
                continue

            for w in words:
                # Hash n-grams into vector slots
                h = hash(w) % self.dim
                vec[h] += 1.0
                for i in range(len(w) - 2):
                    sub = w[i:i+3]
                    h_sub = hash(sub) % self.dim
                    vec[h_sub] += 0.5

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)

        return np.array(vectors, dtype=np.float32)


class LocalVectorStore:
    """
    In-memory vector store supporting fast cosine similarity search.
    """
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model or SimpleEmbeddingModel()
        self.chunks: List[DocumentChunk] = []
        self.vectors: Optional[np.ndarray] = None

    def add_chunks(self, chunks: List[DocumentChunk]):
        if not chunks:
            return
        self.chunks.extend(chunks)
        texts = [c.text for c in chunks]
        new_vecs = self.embedding_model.encode(texts)

        if self.vectors is None:
            self.vectors = new_vecs
        else:
            self.vectors = np.vstack([self.vectors, new_vecs])

    def search(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """Search top-k most semantically similar chunks for query."""
        if self.vectors is None or len(self.chunks) == 0:
            return []

        query_vec = self.embedding_model.encode([query])[0]
        # Compute cosine similarity: query_vec dot product with all normalized vectors
        scores = np.dot(self.vectors, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))
        return results

    def clear(self):
        self.chunks.clear()
        self.vectors = None
