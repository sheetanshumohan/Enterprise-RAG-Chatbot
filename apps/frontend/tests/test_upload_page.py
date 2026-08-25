from unittest.mock import MagicMock, patch
import pytest

from api_client import ApiClient, ApiError


def test_upload_allowed_extensions():
    allowed_types = ["pdf", "docx", "md", "txt"]
    assert "report.pdf".rsplit(".", 1)[-1].lower() in allowed_types
    assert "notes.md".rsplit(".", 1)[-1].lower() in allowed_types
    assert "exe.exe".rsplit(".", 1)[-1].lower() not in allowed_types


@patch("api_client.ApiClient.upload_document")
def test_upload_duplicate_detection(mock_upload):
    mock_upload.side_effect = ApiError(409, "Duplicate document (identical content hash already indexed)")
    client = ApiClient(token="mock-token")

    with pytest.raises(ApiError) as exc_info:
        client.upload_document("col-1", "doc.pdf", b"test content")
    assert exc_info.value.status_code == 409
    assert "Duplicate" in exc_info.value.detail
