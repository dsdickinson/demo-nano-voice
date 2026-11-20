"""
Unit tests for the local RAG engine (ingest, retriever, and prompt augment).
"""

import os
import tempfile
import pytest
from rag.ingest import DocumentIngester, DocumentChunk
from rag.retriever import LocalVectorStore, SimpleEmbeddingModel
from rag.augment import format_rag_prompt


def test_document_ingestion_and_chunking():
    ingester = DocumentIngester(chunk_size=100, chunk_overlap=20)
    text = "The NVIDIA Jetson Orin Nano is an edge AI developer board with 8GB unified memory and Ampere GPU cores."
    chunks = ingester.chunk_text(text, source_file="spec.md")

    assert len(chunks) > 0
    assert all(isinstance(c, DocumentChunk) for c in chunks)
    assert chunks[0].source_file == "spec.md"


def test_vector_store_search():
    store = LocalVectorStore()
    doc1 = DocumentChunk("c1", "Jetson Orin Nano has 8GB unified memory.", "hw.md", 0, 42)
    doc2 = DocumentChunk("c2", "Apples and oranges are nutritious fruit.", "food.md", 0, 40)
    store.add_chunks([doc1, doc2])

    results = store.search("Tell me about the Jetson memory", top_k=1)
    assert len(results) == 1
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "c1"
    assert "Jetson" in top_chunk.text


def test_rag_prompt_augmentation():
    chunk = DocumentChunk("c1", "Nano-Voice runs offline.", "faq.md", 0, 24)
    prompt = format_rag_prompt("Can you run offline?", [(chunk, 0.85)])

    assert "Nano-Voice runs offline." in prompt
    assert "User: Can you run offline?" in prompt
    assert "Nova:" in prompt


def test_public_knowledge_folder_ingestion():
    ingester = DocumentIngester()
    chunks = ingester.ingest_directory("profiles/public/knowledge")
    assert len(chunks) > 0
    # Search against public knowledge
    store = LocalVectorStore()
    store.add_chunks(chunks)
    res = store.search("What is Grouped-Query Attention?", top_k=2)
    assert len(res) > 0
