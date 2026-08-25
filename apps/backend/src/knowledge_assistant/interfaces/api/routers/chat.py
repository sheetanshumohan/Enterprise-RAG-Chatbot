from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from knowledge_assistant.application.use_cases.ask_question import AskQuestionUseCase
from knowledge_assistant.domain.entities import ChatSession, User
from knowledge_assistant.infrastructure.db.repositories import SqlChatRepository
from knowledge_assistant.interfaces.api.dependencies import (
    get_ask_question_use_case,
    get_chat_repo,
    get_current_user,
)
from knowledge_assistant.interfaces.api.schemas import (
    AskRequest,
    ChatSessionCreateRequest,
    ChatSessionResponse,
    ChatSessionUpdateRequest,
    CitationResponse,
    MessageResponse,
)

router = APIRouter(prefix="/chat", tags=["chat"])


def _generate_dynamic_title(question: str) -> str:
    cleaned = question.strip()
    # Strip conversational filler prefixes
    cleaned = re.sub(
        r"^(can you |could you |please |tell me (about )?|what is (the )?|what are (the )?|explain (the )?|how does (the )?|how do (i|we)|compare (the )?|give me (an? )?overview of (the )?)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    # Strip trailing question marks and punctuation
    cleaned = re.sub(r"[\?\.\!\:\;]+$", "", cleaned).strip()
    if not cleaned:
        cleaned = question.strip()

    # Format properly
    if cleaned.islower():
        cleaned = cleaned.title()
    elif len(cleaned) > 0 and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    # Truncate cleanly at word boundary if long
    if len(cleaned) > 36:
        truncated = cleaned[:33].rsplit(" ", 1)[0]
        cleaned = (truncated if len(truncated) > 10 else cleaned[:33]) + "..."

    return cleaned or "Research Discussion"


@router.post("/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: ChatSessionCreateRequest,
    user: User = Depends(get_current_user),
    repo: SqlChatRepository = Depends(get_chat_repo),
):
    session = ChatSession(user_id=user.id, collection_id=payload.collection_id, title=payload.title)
    await repo.create_session(session)
    return ChatSessionResponse(**session.__dict__)


@router.get("/sessions", response_model=list[ChatSessionResponse])
async def list_sessions(user: User = Depends(get_current_user), repo: SqlChatRepository = Depends(get_chat_repo)):
    sessions = await repo.list_sessions(user.id)
    return [ChatSessionResponse(**s.__dict__) for s in sessions]


@router.patch("/sessions/{session_id}", response_model=ChatSessionResponse)
async def update_session_title(
    session_id: str,
    payload: ChatSessionUpdateRequest,
    user: User = Depends(get_current_user),
    repo: SqlChatRepository = Depends(get_chat_repo),
):
    session = await repo.get_session(session_id, user.id)
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    new_title = payload.title.strip()
    await repo.update_title(session_id, new_title)
    session.title = new_title
    return ChatSessionResponse(**session.__dict__)


@router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
async def get_history(
    session_id: str,
    user: User = Depends(get_current_user),
    repo: SqlChatRepository = Depends(get_chat_repo),
):
    session = await repo.get_session(session_id, user.id)
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    messages = await repo.get_history(session_id, limit=200)
    return [
        MessageResponse(
            id=m.id, role=m.role.value, content=m.content,
            citations=[CitationResponse(**c.__dict__) for c in m.citations],
            confidence=m.confidence, reasoning_summary=m.reasoning_summary,
            suggested_followups=m.suggested_followups, created_at=m.created_at,
        )
        for m in messages
    ]


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    user: User = Depends(get_current_user),
    repo: SqlChatRepository = Depends(get_chat_repo),
):
    session = await repo.get_session(session_id, user.id)
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    await repo.delete_session(session_id, user.id)


@router.post("/ask")
async def ask(
    payload: AskRequest,
    user: User = Depends(get_current_user),
    chat_repo: SqlChatRepository = Depends(get_chat_repo),
    ask_use_case: AskQuestionUseCase = Depends(get_ask_question_use_case),
):
    """Server-Sent Events stream of the agentic RAG answer.

    Event payloads (each a JSON-encoded line, `data: {...}\n\n`):
      {"type": "status", "data": "..."}          -- planner/retrieval progress
      {"type": "token", "data": "..."}            -- streamed answer token
      {"type": "final", "data": {citations, confidence, reasoning_summary,
                                  suggested_followups, iterations}}
    """
    session = await chat_repo.get_session(payload.session_id, user.id)
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat session not found")

    # Automatically give the conversation a meaningful dynamic name from the first question
    if session.title in ("New conversation", "Untitled", "", None):
        dynamic_title = _generate_dynamic_title(payload.question)
        await chat_repo.update_title(session.id, dynamic_title)

    async def event_stream():
        async for event in ask_use_case.run_streaming(
            session_id=payload.session_id,
            user_id=user.id,
            collection_id=payload.collection_id or session.collection_id,
            question=payload.question,
            document_ids=payload.document_ids,
        ):
            yield f"data: {json.dumps(_serialize_event(event))}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _serialize_event(value):
    """Recursively convert dataclasses (e.g. Citation) into plain dicts so the
    frontend receives proper JSON objects (`{"document_filename": "..."}`)
    instead of json.dumps' `default=str` fallback silently flattening them
    into opaque strings like "Citation(chunk_id='...', ...)"."""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _serialize_event(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _serialize_event(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_event(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value
