from unittest.mock import MagicMock, patch
import pytest

from api_client import ApiClient


def test_chat_stream_event_aggregation():
    mock_events = [
        {"type": "status", "data": "Planning retrieval..."},
        {"type": "status", "data": "Searching knowledge base..."},
        {"type": "token", "data": "According "},
        {"type": "token", "data": "to the company policy [1], "},
        {"type": "token", "data": "sick leave is 12 days."},
        {
            "type": "final",
            "data": {
                "citations": [
                    {
                        "chunk_id": "c-1",
                        "document_filename": "hr_policy.pdf",
                        "snippet": "Employees get 12 days of paid sick leave per year.",
                        "score": 0.98,
                    }
                ],
                "confidence": 0.92,
                "reasoning_summary": "Direct factual match in hr_policy.pdf.",
                "suggested_followups": ["How do I request sick leave?"],
                "iterations": 1,
            },
        },
    ]

    # Aggregate streamed tokens
    full_answer = ""
    status_updates = []
    final_data = None

    for event in mock_events:
        etype = event["type"]
        if etype == "status":
            status_updates.append(event["data"])
        elif etype == "token":
            full_answer += event["data"]
        elif etype == "final":
            final_data = event["data"]

    assert len(status_updates) == 2
    assert full_answer == "According to the company policy [1], sick leave is 12 days."
    assert final_data is not None
    assert final_data["confidence"] == 0.92
    assert len(final_data["citations"]) == 1
    assert final_data["citations"][0]["document_filename"] == "hr_policy.pdf"
    assert len(final_data["suggested_followups"]) == 1
