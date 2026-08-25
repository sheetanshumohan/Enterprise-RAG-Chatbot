"""Application layer - Document ingestion use case."""
from __future__ import annotations

import logging

from knowledge_assistant.domain.entities import Document
from knowledge_assistant.domain.repositories import ChunkRepository, DocumentRepository, KeywordSearchIndex, VectorStore
from knowledge_assistant.infrastructure.chunking.chunker import build_parent_child_chunks
from knowledge_assistant.infrastructure.chunking.extractors import detect_doc_type, extract_text, sha256_of
from knowledge_assistant.infrastructure.embeddings.client import EmbeddingClient

logger = logging.getLogger(__name__)


class DuplicateDocumentError(Exception):
    def __init__(self, existing_document_id: str):
        self.existing_document_id = existing_document_id
        super().__init__(f"Duplicate of document {existing_document_id}")


class IngestDocumentUseCase:
    def __init__(
        self,
        document_repo: DocumentRepository,
        chunk_repo: ChunkRepository,
        vector_store: VectorStore,
        keyword_index: KeywordSearchIndex,
        embedding_client: EmbeddingClient,
        parent_max_tokens: int = 1500,
        child_max_tokens: int = 300,
        overlap_units: int = 1,
    ) -> None:
        self._document_repo = document_repo
        self._chunk_repo = chunk_repo
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._embedding_client = embedding_client
        self._parent_max_tokens = parent_max_tokens
        self._child_max_tokens = child_max_tokens
        self._overlap_units = overlap_units

    async def execute(
        self,
        user_id: str,
        collection_id: str,
        filename: str,
        raw_bytes: bytes,
        tags: list[str] | None = None,
        allow_reindex_of: str | None = None,
    ) -> Document:
        content_hash = sha256_of(raw_bytes)

        # Duplicate detection
        existing = await self._document_repo.find_by_hash(user_id, content_hash)
        if existing and existing.id != allow_reindex_of:
            raise DuplicateDocumentError(existing.id)

        doc_type = detect_doc_type(filename)
        document = Document(
            user_id=user_id,
            collection_id=collection_id,
            filename=filename,
            doc_type=doc_type,
            content_hash=content_hash,
            tags=tags or [],
        )
        document = await self._document_repo.create(document)

        try:
            document.mark_processing()
            await self._document_repo.update(document)

            text = extract_text(raw_bytes, doc_type)
            chunks = build_parent_child_chunks(
                document,
                text,
                parent_max_tokens=self._parent_max_tokens,
                child_max_tokens=self._child_max_tokens,
                overlap_units=self._overlap_units,
            )
            if not chunks:
                raise ValueError("No extractable text found in document")


            # Only child chunks are embedded/searched directly; parents are context containers.
            child_chunks = [c for c in chunks if c.level.value == "child"]
            texts = [c.text for c in child_chunks]
            embeddings = await self._embedding_client.embed(texts)
            for chunk, embedding in zip(child_chunks, embeddings):
                chunk.embedding = embedding

            await self._chunk_repo.bulk_create(chunks)
            await self._vector_store.upsert(child_chunks)
            await self._keyword_index.index(chunks)

            document.mark_indexed()
            document.metadata["chunk_count"] = len(chunks)
            document.metadata["child_chunk_count"] = len(child_chunks)
            await self._document_repo.update(document)
            return document
        except Exception:
            document.mark_failed()
            await self._document_repo.update(document)
            raise

    async def delete_document(self, document_id: str, user_id: str) -> None:
        document = await self._document_repo.get(document_id, user_id)
        if not document:
            return
        await self._chunk_repo.delete_by_document(document_id)
        try:
            await self._vector_store.delete_by_document(document_id)
        except Exception as exc:
            logger.warning("Vector store deletion skipped/failed for document %s: %s", document_id, exc)
        try:
            await self._keyword_index.delete_by_document(document_id)
        except Exception as exc:
            logger.warning("Keyword index deletion skipped/failed for document %s: %s", document_id, exc)
        await self._document_repo.delete(document_id, user_id)
