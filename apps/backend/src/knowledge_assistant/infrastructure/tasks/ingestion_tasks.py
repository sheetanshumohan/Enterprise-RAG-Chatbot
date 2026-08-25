"""
Async ingestion task.

Each Celery worker process runs its own event loop per task (`asyncio.run`)
and opens a fresh AsyncSession/repo set, rather than sharing the FastAPI
process's engine -- Celery workers and the API server are separate
processes (potentially separate containers), so nothing can be shared
in-memory between them anyway. State that legitimately needs to be
process-wide (the BM25 index, currently in-memory) is the one piece of
this scaffold that would need to move to a shared service (e.g. Redis or
an actual search engine) before running multiple worker replicas in
production -- noted in the README as a known scaling limitation.
"""
from __future__ import annotations

import asyncio
import logging

from knowledge_assistant.application.use_cases.ingest_document import IngestDocumentUseCase
from knowledge_assistant.infrastructure.db.repositories import SqlChunkRepository, SqlDocumentRepository
from knowledge_assistant.infrastructure.db.session import SessionLocal
from knowledge_assistant.infrastructure.embeddings.client import get_embedding_client
from knowledge_assistant.domain.repositories import KeywordSearchIndex
from knowledge_assistant.infrastructure.search.opensearch_index import OpenSearchKeywordIndex
from knowledge_assistant.infrastructure.tasks.celery_app import celery_app
from knowledge_assistant.infrastructure.vector_store.qdrant_store import QdrantVectorStore
from knowledge_assistant.config import settings
from knowledge_assistant.infrastructure.observability.metrics import DOCUMENTS_INGESTED_TOTAL

logger = logging.getLogger(__name__)

# Worker-process-wide singletons (one per worker process, recreated per process).
_vector_store: QdrantVectorStore | None = None
_keyword_index: KeywordSearchIndex | None = None


def _get_vector_store() -> QdrantVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore(
            url=settings.qdrant_url,
            embedding_dim=settings.embedding_dim,
            api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection,
            timeout=settings.qdrant_timeout,
        )
    return _vector_store


def _get_keyword_index() -> KeywordSearchIndex:
    global _keyword_index
    if _keyword_index is None:
        _keyword_index = OpenSearchKeywordIndex(
            hosts=settings.opensearch_url,
            index_name=settings.opensearch_index,
            timeout=settings.opensearch_timeout,
            max_retries=settings.opensearch_max_retries,
        )
    return _keyword_index


async def _ingest_async(user_id: str, collection_id: str, filename: str, raw_bytes: bytes) -> str:
    async with SessionLocal() as session:
        use_case = IngestDocumentUseCase(
            document_repo=SqlDocumentRepository(session),
            chunk_repo=SqlChunkRepository(session),
            vector_store=_get_vector_store(),
            keyword_index=_get_keyword_index(),
            embedding_client=get_embedding_client(
                settings.embedding_provider,
                settings.embedding_api_key,
                model=settings.embedding_model,
                dim=settings.embedding_dim,
            ),
            parent_max_tokens=settings.chunk_parent_max_tokens,
            child_max_tokens=settings.chunk_child_max_tokens,
            overlap_units=settings.chunk_overlap_units,
        )
        document = await use_case.execute(
            user_id=user_id, collection_id=collection_id, filename=filename, raw_bytes=raw_bytes
        )
        return document.id



@celery_app.task(name="ingest_document", bind=True, max_retries=2, default_retry_delay=15)
def ingest_document_task(self, user_id: str, collection_id: str, filename: str, raw_bytes: bytes) -> str:
    try:
        doc_id = asyncio.run(_ingest_async(user_id, collection_id, filename, raw_bytes))
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "unknown"
        DOCUMENTS_INGESTED_TOTAL.labels(status="success", file_type=ext).inc()
        return doc_id
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure should retry/log, not crash the worker
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "unknown"
        DOCUMENTS_INGESTED_TOTAL.labels(status="failed", file_type=ext).inc()
        logger.exception("Ingestion failed for %s (user=%s)", filename, user_id)
        raise self.retry(exc=exc) from exc


