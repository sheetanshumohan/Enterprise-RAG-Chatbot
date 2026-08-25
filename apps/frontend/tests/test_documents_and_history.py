from unittest.mock import MagicMock, patch
import pytest

from api_client import ApiClient


@patch("api_client.ApiClient.list_documents")
def test_filter_documents_by_collection(mock_list):
    mock_list.return_value = [
        {"id": "d1", "filename": "file1.pdf", "collection_id": "col-a", "status": "INDEXED"},
        {"id": "d2", "filename": "file2.pdf", "collection_id": "col-b", "status": "INDEXED"},
    ]
    client = ApiClient(token="mock-token")
    docs = client.list_documents(collection_id="col-a")
    mock_list.assert_called_once_with(collection_id="col-a")
    assert len(docs) == 2


@patch("api_client.ApiClient.get_history")
def test_conversation_history_retrieval(mock_history):
    mock_history.return_value = [
        {"id": "m1", "role": "user", "content": "Hello", "citations": []},
        {"id": "m2", "role": "assistant", "content": "Hi! How can I help?", "citations": []},
    ]
    client = ApiClient(token="mock-token")
    history = client.get_history("session-123")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
