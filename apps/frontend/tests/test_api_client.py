import json
from unittest.mock import MagicMock, patch
import httpx
import pytest

from api_client import ApiClient, ApiError


def test_api_error_str():
    err = ApiError(404, "Collection not found")
    assert err.status_code == 404
    assert err.detail == "Collection not found"
    assert "404: Collection not found" in str(err)


def test_api_client_headers():
    client_no_auth = ApiClient()
    assert client_no_auth._headers() == {}

    client_with_auth = ApiClient(token="jwt-test-token-123")
    assert client_with_auth._headers() == {"Authorization": "Bearer jwt-test-token-123"}


def test_api_client_handle_success():
    client = ApiClient()
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.content = b'{"status": "ok"}'
    mock_resp.json.return_value = {"status": "ok"}

    assert client._handle(mock_resp) == {"status": "ok"}


def test_api_client_handle_empty_204():
    client = ApiClient()
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 204
    mock_resp.content = b""

    assert client._handle(mock_resp) is None


def test_api_client_handle_error_json():
    client = ApiClient()
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 400
    mock_resp.content = b'{"detail": "Invalid credentials"}'
    mock_resp.json.return_value = {"detail": "Invalid credentials"}

    with pytest.raises(ApiError) as exc_info:
        client._handle(mock_resp)
    assert exc_info.value.status_code == 400
    assert "Invalid credentials" in exc_info.value.detail


def test_api_client_handle_error_plain_text():
    client = ApiClient()
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 502
    mock_resp.content = b"Bad Gateway"
    mock_resp.json.side_effect = ValueError("Not JSON")
    mock_resp.text = "Bad Gateway"

    with pytest.raises(ApiError) as exc_info:
        client._handle(mock_resp)
    assert exc_info.value.status_code == 502
    assert "Bad Gateway" in exc_info.value.detail


@patch("httpx.post")
def test_login_and_register(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{"access_token": "token-xyz"}'
    mock_resp.json.return_value = {"access_token": "token-xyz"}
    mock_post.return_value = mock_resp

    client = ApiClient()
    token = client.login("user@example.com", "secret123")
    assert token == "token-xyz"
    mock_post.assert_called_with(
        "http://localhost:8000/auth/login",
        json={"email": "user@example.com", "password": "secret123"},
        timeout=120.0,
    )

    reg_token = client.register("user@example.com", "secret123", "Alice")
    assert reg_token == "token-xyz"


@patch("httpx.get")
def test_me(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{"id": "u1", "email": "user@example.com"}'
    mock_resp.json.return_value = {"id": "u1", "email": "user@example.com"}
    mock_get.return_value = mock_resp

    client = ApiClient(token="valid-token")
    user = client.me()
    assert user["id"] == "u1"
    mock_get.assert_called_with(
        "http://localhost:8000/auth/me",
        headers={"Authorization": "Bearer valid-token"},
        timeout=120.0,
    )


@patch("httpx.get")
@patch("httpx.post")
@patch("httpx.delete")
def test_collection_endpoints(mock_delete, mock_post, mock_get):
    client = ApiClient(token="test-token")

    # list collections
    mock_get.return_value = MagicMock(
        status_code=200,
        content=b'[{"id": "c1", "name": "Work"}]',
        json=lambda: [{"id": "c1", "name": "Work"}],
    )
    cols = client.list_collections()
    assert len(cols) == 1
    assert cols[0]["name"] == "Work"

    # create collection
    mock_post.return_value = MagicMock(
        status_code=201,
        content=b'{"id": "c2", "name": "Research"}',
        json=lambda: {"id": "c2", "name": "Research"},
    )
    new_col = client.create_collection("Research", "ML papers")
    assert new_col["id"] == "c2"

    # delete collection
    mock_delete.return_value = MagicMock(status_code=204, content=b"")
    client.delete_collection("c1")
    mock_delete.assert_called_with(
        "http://localhost:8000/collections/c1",
        headers={"Authorization": "Bearer test-token"},
        timeout=120.0,
    )


@patch("httpx.get")
@patch("httpx.post")
@patch("httpx.delete")
def test_document_endpoints(mock_delete, mock_post, mock_get):
    client = ApiClient(token="test-token")

    # list documents
    mock_get.return_value = MagicMock(
        status_code=200,
        content=b'[{"id": "d1", "filename": "report.pdf"}]',
        json=lambda: [{"id": "d1", "filename": "report.pdf"}],
    )
    docs = client.list_documents(collection_id="c1")
    assert len(docs) == 1

    # upload document
    mock_post.return_value = MagicMock(
        status_code=202,
        content=b'{"id": "d2", "filename": "paper.pdf", "status": "PENDING"}',
        json=lambda: {"id": "d2", "filename": "paper.pdf", "status": "PENDING"},
    )
    uploaded = client.upload_document("c1", "paper.pdf", b"pdf data")
    assert uploaded["id"] == "d2"

    # delete document
    mock_delete.return_value = MagicMock(status_code=204, content=b"")
    client.delete_document("d1")
    mock_delete.assert_called_with(
        "http://localhost:8000/documents/d1",
        headers={"Authorization": "Bearer test-token"},
        timeout=120.0,
    )


@patch("httpx.stream")
def test_ask_stream(mock_stream):
    client = ApiClient(token="test-token")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = [
        "data: {\"type\": \"status\", \"data\": \"Searching documents...\"}",
        "data: {\"type\": \"token\", \"data\": \"Hello\"}",
        "data: {\"type\": \"token\", \"data\": \" world\"}",
        "",
        "data: {\"type\": \"final\", \"data\": {\"confidence\": 0.95}}",
    ]

    mock_stream.return_value.__enter__.return_value = mock_response

    events = list(client.ask_stream("sess-1", "col-1", "What is RAG?"))
    assert len(events) == 4
    assert events[0] == {"type": "status", "data": "Searching documents..."}
    assert events[1] == {"type": "token", "data": "Hello"}
    assert events[2] == {"type": "token", "data": " world"}
    assert events[3]["data"]["confidence"] == 0.95
