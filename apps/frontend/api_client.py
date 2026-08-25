"""
Thin HTTP client wrapping the FastAPI backend.

All business logic (retrieval, chunking, RAG orchestration, auth) lives in
the backend, per the project's architecture rule: Streamlit is presentation
only. This module's only job is translating Streamlit calls into HTTP
requests against the API and shaping responses for display.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import httpx

try:
    from zoneinfo import ZoneInfo
    ASIA_TZ = ZoneInfo("Asia/Kolkata")
except Exception:
    ASIA_TZ = timezone(timedelta(hours=5, minutes=30))


def format_asia_time(
    ts_str_or_dt: str | datetime | None,
    format_pattern: str = "%b %d, %Y · %I:%M %p IST",
) -> str:
    """Converts a UTC or naive timestamp to Asia/Kolkata (IST, UTC+5:30) and formats it."""
    if not ts_str_or_dt:
        return ""
    try:
        if isinstance(ts_str_or_dt, str):
            clean_ts = ts_str_or_dt.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_ts)
        else:
            dt = ts_str_or_dt

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        asia_dt = dt.astimezone(ASIA_TZ)
        return asia_dt.strftime(format_pattern)
    except Exception:
        return str(ts_str_or_dt)


API_URL = os.environ.get("KNOWLEDGE_ASSISTANT_API_URL", "http://localhost:8000")
API_TIMEOUT = float(os.environ.get("KNOWLEDGE_ASSISTANT_API_TIMEOUT", "120.0"))


class ApiError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{status_code}: {detail}")


class ApiClient:
    def __init__(self, token: str | None = None, timeout: float = API_TIMEOUT):
        self.token = token
        self.timeout = timeout

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _handle(self, response: httpx.Response):
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise ApiError(response.status_code, str(detail))
        return response.json() if response.content else None

    # --- auth ---

    def register(self, email: str, password: str, full_name: str = "") -> str:
        r = httpx.post(
            f"{API_URL}/auth/register",
            json={"email": email, "password": password, "full_name": full_name},
            timeout=self.timeout,
        )
        return self._handle(r)["access_token"]

    def login(self, email: str, password: str) -> str:
        r = httpx.post(
            f"{API_URL}/auth/login",
            json={"email": email, "password": password},
            timeout=self.timeout,
        )
        return self._handle(r)["access_token"]

    def me(self) -> dict:
        r = httpx.get(f"{API_URL}/auth/me", headers=self._headers(), timeout=self.timeout)
        return self._handle(r)

    # --- collections ---

    def list_collections(self) -> list[dict]:
        r = httpx.get(f"{API_URL}/collections", headers=self._headers(), timeout=self.timeout)
        return self._handle(r)

    def create_collection(self, name: str, description: str = "") -> dict:
        r = httpx.post(
            f"{API_URL}/collections",
            json={"name": name, "description": description},
            headers=self._headers(),
            timeout=self.timeout,
        )
        return self._handle(r)

    def delete_collection(self, collection_id: str) -> None:
        r = httpx.delete(f"{API_URL}/collections/{collection_id}", headers=self._headers(), timeout=self.timeout)
        self._handle(r)

    # --- documents ---

    def list_documents(self, collection_id: str | None = None) -> list[dict]:
        params = {"collection_id": collection_id} if collection_id else {}
        r = httpx.get(f"{API_URL}/documents", params=params, headers=self._headers(), timeout=self.timeout)
        return self._handle(r)

    def upload_document(
        self, collection_id: str, filename: str, file_bytes: bytes, tags: str | None = None
    ) -> dict:
        params = {"collection_id": collection_id}
        if tags:
            params["tags"] = tags
        r = httpx.post(
            f"{API_URL}/documents/upload",
            params=params,
            files={"file": (filename, file_bytes)},
            headers=self._headers(),
            timeout=self.timeout,
        )
        return self._handle(r)

    def upload_documents_batch(
        self, collection_id: str, files: list[tuple[str, bytes]], tags: str | None = None
    ) -> dict:
        params = {"collection_id": collection_id}
        if tags:
            params["tags"] = tags
        file_payload = [("files", (fname, fbytes)) for fname, fbytes in files]
        r = httpx.post(
            f"{API_URL}/documents/upload-batch",
            params=params,
            files=file_payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        return self._handle(r)

    def import_arxiv(
        self, collection_id: str, arxiv_id_or_url: str, tags: list[str] | None = None
    ) -> dict:
        payload = {
            "collection_id": collection_id,
            "arxiv_id_or_url": arxiv_id_or_url,
            "tags": tags or ["arxiv", "research-paper"],
        }
        r = httpx.post(
            f"{API_URL}/documents/import-arxiv",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        return self._handle(r)

    def upload_document_async(
        self, collection_id: str, filename: str, file_bytes: bytes
    ) -> dict:
        r = httpx.post(
            f"{API_URL}/documents/upload-async",
            params={"collection_id": collection_id},
            files={"file": (filename, file_bytes)},
            headers=self._headers(),
            timeout=self.timeout,
        )
        return self._handle(r)

    def delete_document(self, document_id: str) -> None:
        r = httpx.delete(f"{API_URL}/documents/{document_id}", headers=self._headers(), timeout=self.timeout)
        self._handle(r)

    # --- chat ---

    def list_sessions(self) -> list[dict]:
        r = httpx.get(f"{API_URL}/chat/sessions", headers=self._headers(), timeout=self.timeout)
        return self._handle(r)

    def create_session(self, collection_id: str | None, title: str = "New conversation") -> dict:
        r = httpx.post(
            f"{API_URL}/chat/sessions",
            json={"collection_id": collection_id, "title": title},
            headers=self._headers(),
            timeout=self.timeout,
        )
        return self._handle(r)

    def get_history(self, session_id: str) -> list[dict]:
        r = httpx.get(f"{API_URL}/chat/sessions/{session_id}/messages", headers=self._headers(), timeout=self.timeout)
        return self._handle(r)

    def delete_session(self, session_id: str) -> None:
        r = httpx.delete(f"{API_URL}/chat/sessions/{session_id}", headers=self._headers(), timeout=self.timeout)
        self._handle(r)

    def update_session_title(self, session_id: str, title: str) -> dict:
        r = httpx.patch(
            f"{API_URL}/chat/sessions/{session_id}",
            json={"title": title},
            headers=self._headers(),
            timeout=self.timeout,
        )
        return self._handle(r)

    def ask_stream(
        self,
        session_id: str,
        collection_id: str | None,
        question: str,
        document_ids: list[str] | None = None,
    ) -> Iterator[dict]:
        """Yields parsed SSE event dicts as they stream in from the backend."""
        payload = {
            "session_id": session_id,
            "collection_id": collection_id,
            "document_ids": document_ids,
            "question": question,
        }
        with httpx.stream(
            "POST", f"{API_URL}/chat/ask", json=payload, headers=self._headers(), timeout=self.timeout
        ) as response:
            if response.status_code >= 400:
                response.read()
                self._handle(response)
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[len("data: "):])


def init_auth() -> ApiClient | None:
    """
    Initializes and preserves authentication state across page refreshes and multipage navigation.
    Uses st.session_state and st.query_params to persist the JWT token.
    """
    try:
        # pyrefly: ignore [missing-import]
        import streamlit as st
    except ImportError:
        return None

    token = st.session_state.get("token")
    if not token and hasattr(st, "query_params") and "auth" in st.query_params:
        auth_param = st.query_params.get("auth")
        if isinstance(auth_param, list):
            auth_param = auth_param[0] if auth_param else None
        if auth_param:
            try:
                user = ApiClient(token=auth_param).me()
                st.session_state["token"] = auth_param
                st.session_state["user"] = user
                token = auth_param
            except Exception:
                if hasattr(st, "query_params"):
                    st.query_params.pop("auth", None)

    if token and hasattr(st, "query_params") and "auth" not in st.query_params:
        st.query_params["auth"] = token

    return ApiClient(token=token) if token else None


def logout() -> None:
    """Clears authentication state and URL query params."""
    try:
        import streamlit as st
        st.session_state["token"] = None
        st.session_state["user"] = None
        st.session_state["active_session_id"] = None
        st.session_state["active_collection_id"] = None
        if hasattr(st, "query_params"):
            st.query_params.pop("auth", None)
    except Exception:
        pass


