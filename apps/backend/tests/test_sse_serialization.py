"""Regression test for a real bug: json.dumps(event, default=str) was
silently flattening Citation dataclasses into opaque strings instead of
proper JSON objects, breaking the frontend's citation['document_filename']
access. This test locks in the fix in interfaces/api/routers/chat.py."""
from __future__ import annotations

import json
from dataclasses import dataclass

from knowledge_assistant.domain.entities import Citation
from knowledge_assistant.interfaces.api.routers.chat import _serialize_event


def test_citation_serializes_to_a_proper_dict_not_a_string():
    citation = Citation(
        chunk_id="c1", document_id="d1", document_filename="policy.txt",
        snippet="Employees get 12 sick days.", score=0.87,
    )
    event = {"type": "final", "data": {"citations": [citation], "confidence": 0.9}}

    serialized = _serialize_event(event)
    round_tripped = json.loads(json.dumps(serialized))

    citation_out = round_tripped["data"]["citations"][0]
    assert isinstance(citation_out, dict), "citation must serialize to a JSON object, not a string"
    assert citation_out["document_filename"] == "policy.txt"
    assert citation_out["score"] == 0.87


def test_nested_dataclasses_and_plain_values_all_serialize():
    @dataclass
    class Inner:
        x: int

    @dataclass
    class Outer:
        inner: Inner
        label: str

    value = {"items": [Outer(inner=Inner(x=1), label="a"), "plain string", 42, None]}
    serialized = _serialize_event(value)
    json.dumps(serialized)  # must not raise

    assert serialized["items"][0] == {"inner": {"x": 1}, "label": "a"}
    assert serialized["items"][1] == "plain string"
    assert serialized["items"][2] == 42
    assert serialized["items"][3] is None
