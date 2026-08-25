from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str | None


class CollectionCreateRequest(BaseModel):
    name: str
    description: str = ""


class CollectionResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime


class DocumentResponse(BaseModel):
    id: str
    filename: str
    doc_type: str
    status: str
    version: int
    tags: list[str]
    metadata: dict
    created_at: datetime
    updated_at: datetime


class ArxivImportRequest(BaseModel):
    collection_id: str
    arxiv_id_or_url: str
    tags: list[str] = ["arxiv", "research-paper"]


class BatchUploadItemResult(BaseModel):
    filename: str
    status: str
    document_id: str | None = None
    chunk_count: int | None = None
    error: str | None = None


class BatchUploadResponse(BaseModel):
    total: int
    successful: int
    failed: int
    results: list[BatchUploadItemResult]


class ChatSessionCreateRequest(BaseModel):
    collection_id: str | None = None
    title: str = "New conversation"


class ChatSessionUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class ChatSessionResponse(BaseModel):
    id: str
    collection_id: str | None
    title: str
    created_at: datetime


class AskRequest(BaseModel):
    session_id: str
    collection_id: str | None = None
    document_ids: list[str] | None = None
    question: str


class CitationResponse(BaseModel):
    chunk_id: str
    document_id: str
    document_filename: str
    snippet: str
    score: float


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    citations: list[CitationResponse]
    confidence: float | None
    reasoning_summary: str | None
    suggested_followups: list[str]
    created_at: datetime
