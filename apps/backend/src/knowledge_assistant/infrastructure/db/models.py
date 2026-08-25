from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class CollectionModel(Base):
    __tablename__ = "collections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    collection_id: Mapped[str] = mapped_column(String(36), ForeignKey("collections.id"), index=True)
    filename: Mapped[str] = mapped_column(String(512))
    doc_type: Mapped[str] = mapped_column(String(32))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32))
    version: Mapped[int] = mapped_column(Integer, default=1)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    doc_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ChunkModel(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    collection_id: Mapped[str] = mapped_column(String(36), index=True)
    text: Mapped[str] = mapped_column(Text)
    level: Mapped[str] = mapped_column(String(16))
    parent_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    position: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    # embeddings themselves live in Qdrant, not Postgres


class ChatSessionModel(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    collection_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    title: Mapped[str] = mapped_column(String(255), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MessageModel(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("chat_sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reasoning_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    suggested_followups: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RetrievalLogModel(Base):
    __tablename__ = "retrieval_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    rewritten_queries: Mapped[list] = mapped_column(JSON, default=list)
    retriever_used: Mapped[str] = mapped_column(String(64))
    iterations: Mapped[int] = mapped_column(Integer, default=1)
    chunks_retrieved: Mapped[int] = mapped_column(Integer, default=0)
    retrieval_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    llm_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    embedding_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class FeedbackModel(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    message_id: Mapped[str] = mapped_column(String(36), ForeignKey("messages.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    rating: Mapped[int] = mapped_column(Integer)
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class EvaluationResultModel(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    precision: Mapped[float] = mapped_column(Float, default=0.0)
    recall: Mapped[float] = mapped_column(Float, default=0.0)
    groundedness: Mapped[float] = mapped_column(Float, default=0.0)
    context_precision: Mapped[float] = mapped_column(Float, default=0.0)
    answer_relevance: Mapped[float] = mapped_column(Float, default=0.0)
    hallucination_rate: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
