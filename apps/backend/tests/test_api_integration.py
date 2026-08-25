"""
API integration tests: exercise the real FastAPI app over HTTP (via ASGI
transport, no network socket needed) against a throwaway SQLite DB, proving
the full DI graph, auth flow, and per-user isolation actually work -- not
just the individual layers in isolation.

Requires: httpx, aiosqlite, asgi-lifespan (dev dependencies).
Run with: DATABASE_URL=sqlite+aiosqlite:///:memory: pytest tests/test_api_integration.py
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_api.db")

pytest.importorskip("aiosqlite")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from knowledge_assistant.infrastructure.db.models import Base  # noqa: E402
from knowledge_assistant.infrastructure.db.session import engine, init_db  # noqa: E402
from knowledge_assistant.interfaces.api.main import app  # noqa: E402


@pytest.fixture
async def client():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _register(client, email: str, password: str = "password123") -> dict:
    r = await client.post("/auth/register", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    return r.json()


@pytest.mark.asyncio
async def test_register_and_login_flow(client):
    reg = await _register(client, "alice@example.com")
    assert "access_token" in reg

    login = await client.post("/auth/login", json={"email": "alice@example.com", "password": "password123"})
    assert login.status_code == 200
    assert "access_token" in login.json()


@pytest.mark.asyncio
async def test_login_with_wrong_password_is_rejected(client):
    await _register(client, "bob@example.com")
    login = await client.post("/auth/login", json={"email": "bob@example.com", "password": "wrong-password"})
    assert login.status_code == 401


@pytest.mark.asyncio
async def test_duplicate_registration_is_rejected(client):
    await _register(client, "carol@example.com")
    dup = await client.post("/auth/register", json={"email": "carol@example.com", "password": "password123"})
    assert dup.status_code == 409


@pytest.mark.asyncio
async def test_protected_route_requires_auth(client):
    r = await client.get("/collections")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_collection_crud_and_user_isolation(client):
    alice = await _register(client, "alice2@example.com")
    bob = await _register(client, "bob2@example.com")
    alice_headers = {"Authorization": f"Bearer {alice['access_token']}"}
    bob_headers = {"Authorization": f"Bearer {bob['access_token']}"}

    create = await client.post(
        "/collections", json={"name": "Research Papers", "description": "papers"}, headers=alice_headers
    )
    assert create.status_code == 201
    collection_id = create.json()["id"]

    alice_list = await client.get("/collections", headers=alice_headers)
    assert len(alice_list.json()) == 1

    # Bob must never see Alice's collections
    bob_list = await client.get("/collections", headers=bob_headers)
    assert bob_list.json() == []

    # Bob cannot delete Alice's collection (route treats it as not found for him)
    delete_attempt = await client.delete(f"/collections/{collection_id}", headers=bob_headers)
    assert delete_attempt.status_code in (204, 404)  # deletes are no-ops for non-owners, not 403 leaks
    still_there = await client.get("/collections", headers=alice_headers)
    assert len(still_there.json()) == 1  # Alice's collection is untouched


@pytest.mark.asyncio
async def test_health_check(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
