"""
Parent-child chunking.

Strategy:
  1. Split the cleaned document text into large "parent" sections
     (~1500 tokens, paragraph-aligned) that preserve broad context.
  2. Split each parent into small "child" chunks (~300 tokens, sentence
     aligned) that are what actually gets embedded and searched.

At query time we search over CHILD chunks (precise matching) but expand
each hit to its PARENT chunk before handing it to the LLM (better
context, fewer "lost in the middle" truncation artifacts). This is the
standard parent-document-retriever pattern.
"""
from __future__ import annotations

import re

from knowledge_assistant.domain.entities import Chunk, ChunkLevel, Document

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:  # pragma: no cover - fallback if tiktoken data unavailable offline
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_sentences(text: str) -> list[str]:
    # Lightweight sentence splitter; avoids pulling in a heavy NLP dependency.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", text)
    return [s.strip() for s in sentences if s.strip()]


def _pack(units: list[str], max_tokens: int, overlap_units: int = 1) -> list[str]:
    """Greedily pack text units (paragraphs or sentences) into chunks under max_tokens,
    keeping a small overlap between consecutive chunks for retrieval continuity."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit)
        if current and current_tokens + unit_tokens > max_tokens:
            chunks.append(" ".join(current))
            current = current[-overlap_units:] if overlap_units else []
            current_tokens = sum(count_tokens(u) for u in current)
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


def build_parent_child_chunks(
    document: Document,
    text: str,
    parent_max_tokens: int = 1500,
    child_max_tokens: int = 300,
    overlap_units: int = 1,
) -> list[Chunk]:
    """Returns a flat list of Chunk entities: PARENT chunks followed by their CHILD chunks."""
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    parent_texts = _pack(paragraphs, parent_max_tokens, overlap_units=overlap_units)

    all_chunks: list[Chunk] = []
    for p_idx, parent_text in enumerate(parent_texts):
        parent = Chunk(
            document_id=document.id,
            user_id=document.user_id,
            collection_id=document.collection_id,
            text=parent_text,
            level=ChunkLevel.PARENT,
            position=p_idx,
            token_count=count_tokens(parent_text),
            metadata={"filename": document.filename, "doc_type": document.doc_type.value},
        )
        all_chunks.append(parent)

        sentences = _split_sentences(parent_text)
        child_texts = _pack(sentences, child_max_tokens, overlap_units=overlap_units)

        for c_idx, child_text in enumerate(child_texts):
            child = Chunk(
                document_id=document.id,
                user_id=document.user_id,
                collection_id=document.collection_id,
                text=child_text,
                level=ChunkLevel.CHILD,
                parent_id=parent.id,
                position=c_idx,
                token_count=count_tokens(child_text),
                metadata={"filename": document.filename, "doc_type": document.doc_type.value},
            )
            all_chunks.append(child)

    return all_chunks
