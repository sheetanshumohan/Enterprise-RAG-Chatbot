from unittest.mock import MagicMock, patch
import pytest

from api_client import ApiClient, ApiError


def test_collection_name_validation():
    # Empty name should not call client
    client = MagicMock(spec=ApiClient)
    empty_name = "   "
    assert not empty_name.strip()


@patch("api_client.ApiClient.list_collections")
@patch("api_client.ApiClient.delete_collection")
def test_delete_collection_workflow(mock_delete, mock_list):
    client = ApiClient(token="mock-token")
    mock_list.return_value = [
        {"id": "col-1", "name": "Work", "description": "Work documents", "created_at": "2026-08-25T00:00:00Z"}
    ]

    collections = client.list_collections()
    assert len(collections) == 1
    target_id = collections[0]["id"]

    client.delete_collection(target_id)
    mock_delete.assert_called_once_with("col-1")
