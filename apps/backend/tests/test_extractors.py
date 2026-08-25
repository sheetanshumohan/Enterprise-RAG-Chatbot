import pytest

from knowledge_assistant.domain.entities import DocumentType
from knowledge_assistant.infrastructure.chunking.extractors import detect_doc_type, extract_text, sha256_of


def test_detect_doc_type_maps_extensions():
    assert detect_doc_type("report.pdf") == DocumentType.PDF
    assert detect_doc_type("notes.docx") == DocumentType.DOCX
    assert detect_doc_type("readme.md") == DocumentType.MARKDOWN
    assert detect_doc_type("plain.txt") == DocumentType.TXT


def test_detect_doc_type_rejects_unsupported_extension():
    with pytest.raises(ValueError):
        detect_doc_type("archive.zip")


def test_extract_text_txt_cleans_whitespace():
    raw = b"Line one.\r\n\r\n\r\n\r\nLine two.   with   extra   spaces."
    text = extract_text(raw, DocumentType.TXT)
    assert "\n\n\n" not in text
    assert "Line one." in text
    assert "Line two." in text


def test_sha256_is_deterministic_and_sensitive_to_content():
    a = sha256_of(b"hello world")
    b = sha256_of(b"hello world")
    c = sha256_of(b"hello world!")
    assert a == b
    assert a != c
    assert len(a) == 64
