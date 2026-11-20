"""
Prompt augmentation utility for grounding the LLM with retrieved context.
"""

from typing import List, Tuple
from rag.ingest import DocumentChunk


DEFAULT_SYSTEM_PROMPT = (
    "You are Nova, an intelligent, concise, articulate edge AI assistant with a natural female persona. "
    "Use the provided context to answer the user's question directly and concisely."
)


def format_rag_prompt(
    query: str,
    retrieved_chunks: List[Tuple[DocumentChunk, float]],
    persona_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_context_chars: int = 1000
) -> str:
    """
    Format user query and retrieved context chunks into an augmented LLM prompt.
    """
    context_str = ""
    current_len = 0

    for chunk, score in retrieved_chunks:
        if score < 0.15:  # skip completely irrelevant matches
            continue
        snippet = f"- {chunk.text.strip()}\n"
        if current_len + len(snippet) > max_context_chars:
            break
        context_str += snippet
        current_len += len(snippet)

    if context_str.strip():
        prompt = (
            f"<system>\n{persona_prompt}\n</system>\n"
            f"<context>\n{context_str.strip()}\n</context>\n"
            f"User: {query}\n"
            f"Nova:"
        )
    else:
        prompt = (
            f"<system>\n{persona_prompt}\n</system>\n"
            f"User: {query}\n"
            f"Nova:"
        )

    return prompt
