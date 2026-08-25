"""
Composition root.

This is the ONLY place in the codebase that is allowed to import concrete
infrastructure classes and wire them to abstract ports and application
use cases. Routers depend on these `Depends(...)` functions, never on
infrastructure classes directly.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_assistant.application.use_cases.ask_question import AskQuestionUseCase
from knowledge_assistant.application.use_cases.ingest_document import IngestDocumentUseCase
from knowledge_assistant.application.use_cases.retrieve_context import HybridRetriever
from knowledge_assistant.config import settings
from knowledge_assistant.domain.entities import User
from knowledge_assistant.infrastructure.auth.jwt_auth import decode_access_token
from knowledge_assistant.infrastructure.db.repositories import (
    SqlChatRepository,
    SqlChunkRepository,
    SqlCollectionRepository,
    SqlDocumentRepository,
    SqlEvaluationRepository,
    SqlFeedbackRepository,
    SqlRetrievalLogRepository,
    SqlUserRepository,
)
from knowledge_assistant.infrastructure.db.session import get_session
from knowledge_assistant.infrastructure.embeddings.client import get_embedding_client
from knowledge_assistant.infrastructure.llm.client import get_llm_client
from knowledge_assistant.domain.repositories import KeywordSearchIndex
from knowledge_assistant.infrastructure.reranking.reranker import get_reranker
from knowledge_assistant.infrastructure.search.opensearch_index import OpenSearchKeywordIndex
from knowledge_assistant.infrastructure.vector_store.qdrant_store import QdrantVectorStore


# --- process-wide singletons (stateful clients, safe to share across requests) ---

@lru_cache
def get_embedding_client_singleton():
    return get_embedding_client(
        settings.embedding_provider,
        settings.embedding_api_key,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
    )


@lru_cache
def get_llm_client_singleton():
    return get_llm_client(settings.llm_provider, settings.llm_api_key, settings.llm_model)


@lru_cache
def get_vector_store_singleton() -> QdrantVectorStore:
    return QdrantVectorStore(
        url=settings.qdrant_url,
        embedding_dim=settings.embedding_dim,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
        timeout=settings.qdrant_timeout,
    )


@lru_cache
def get_keyword_index_singleton() -> KeywordSearchIndex:
    return OpenSearchKeywordIndex(
        hosts=settings.opensearch_url,
        index_name=settings.opensearch_index,
        timeout=settings.opensearch_timeout,
        max_retries=settings.opensearch_max_retries,
    )


@lru_cache
def get_reranker_singleton():
    return get_reranker(
        use_cross_encoder=settings.use_cross_encoder_reranker,
        model_name=settings.cross_encoder_model,
    )


# --- per-request repositories (bound to the request's DB session) ---

def get_user_repo(session: AsyncSession = Depends(get_session)) -> SqlUserRepository:
    return SqlUserRepository(session)


def get_collection_repo(session: AsyncSession = Depends(get_session)) -> SqlCollectionRepository:
    return SqlCollectionRepository(session)


def get_document_repo(session: AsyncSession = Depends(get_session)) -> SqlDocumentRepository:
    return SqlDocumentRepository(session)


def get_chunk_repo(session: AsyncSession = Depends(get_session)) -> SqlChunkRepository:
    return SqlChunkRepository(session)


def get_chat_repo(session: AsyncSession = Depends(get_session)) -> SqlChatRepository:
    return SqlChatRepository(session)


def get_retrieval_log_repo(session: AsyncSession = Depends(get_session)) -> SqlRetrievalLogRepository:
    return SqlRetrievalLogRepository(session)


def get_feedback_repo(session: AsyncSession = Depends(get_session)) -> SqlFeedbackRepository:
    return SqlFeedbackRepository(session)


def get_evaluation_repo(session: AsyncSession = Depends(get_session)) -> SqlEvaluationRepository:
    return SqlEvaluationRepository(session)


# --- application use cases (assembled from the above ports) ---

def get_ingest_use_case(
    document_repo: SqlDocumentRepository = Depends(get_document_repo),
    chunk_repo: SqlChunkRepository = Depends(get_chunk_repo),
    vector_store: QdrantVectorStore = Depends(get_vector_store_singleton),
    keyword_index: KeywordSearchIndex = Depends(get_keyword_index_singleton),
    embedding_client = Depends(get_embedding_client_singleton),
) -> IngestDocumentUseCase:
    return IngestDocumentUseCase(
        document_repo=document_repo,
        chunk_repo=chunk_repo,
        vector_store=vector_store,
        keyword_index=keyword_index,
        embedding_client=embedding_client,
        parent_max_tokens=settings.chunk_parent_max_tokens,
        child_max_tokens=settings.chunk_child_max_tokens,
        overlap_units=settings.chunk_overlap_units,
    )


def get_hybrid_retriever(
    chunk_repo: SqlChunkRepository = Depends(get_chunk_repo),
    vector_store: QdrantVectorStore = Depends(get_vector_store_singleton),
    keyword_index: KeywordSearchIndex = Depends(get_keyword_index_singleton),
    embedding_client = Depends(get_embedding_client_singleton),
    reranker = Depends(get_reranker_singleton),
) -> HybridRetriever:
    return HybridRetriever(
        vector_store=vector_store,
        keyword_index=keyword_index,
        chunk_repo=chunk_repo,
        embedding_client=embedding_client,
        reranker=reranker,
    )


def get_ask_question_use_case(
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    chat_repo: SqlChatRepository = Depends(get_chat_repo),
    retrieval_log_repo: SqlRetrievalLogRepository = Depends(get_retrieval_log_repo),
    llm = Depends(get_llm_client_singleton),
) -> AskQuestionUseCase:
    return AskQuestionUseCase(
        retriever=retriever,
        llm=llm,
        chat_repo=chat_repo,
        retrieval_log_repo=retrieval_log_repo,
        max_retrieval_iterations=settings.rag_max_retrieval_iterations,
        max_generation_tokens=settings.rag_max_generation_tokens,
        planner_max_tokens=settings.rag_planner_max_tokens,
        evaluator_max_tokens=settings.rag_evaluator_max_tokens,
        followup_max_tokens=settings.rag_followup_max_tokens,
        history_messages_limit=settings.rag_history_messages_limit,
    )



# --- auth dependency ---

async def get_current_user(
    authorization: str = Header(default=""),
    user_repo: SqlUserRepository = Depends(get_user_repo),
) -> User:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    user_id = decode_access_token(token)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    user = await user_repo.get_by_id(user_id)
    if not user or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found or inactive")
    return user
