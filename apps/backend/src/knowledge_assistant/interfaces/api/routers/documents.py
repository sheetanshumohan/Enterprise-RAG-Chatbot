from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from knowledge_assistant.application.use_cases.ingest_document import DuplicateDocumentError, IngestDocumentUseCase
from knowledge_assistant.config import settings
from knowledge_assistant.domain.entities import User
from knowledge_assistant.infrastructure.db.repositories import SqlDocumentRepository
from knowledge_assistant.interfaces.api.dependencies import (
    get_current_user,
    get_document_repo,
    get_ingest_use_case,
)
from knowledge_assistant.interfaces.api.schemas import (
    ArxivImportRequest,
    BatchUploadItemResult,
    BatchUploadResponse,
    DocumentResponse,
)

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {"pdf", "docx", "md", "markdown", "txt"}


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    collection_id: str,
    file: UploadFile,
    tags: str | None = None,
    user: User = Depends(get_current_user),
    ingest: IngestDocumentUseCase = Depends(get_ingest_use_case),
):
    safe_filename = file.filename or "uploaded_file"
    ext = safe_filename.lower().rsplit(".", 1)[-1] if "." in safe_filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported file type: .{ext}")

    raw_bytes = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {settings.max_upload_mb}MB limit")

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    try:
        document = await ingest.execute(
            user_id=user.id,
            collection_id=collection_id,
            filename=safe_filename,
            raw_bytes=raw_bytes,
            tags=tag_list,
        )
    except DuplicateDocumentError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"This file was already uploaded (document_id={exc.existing_document_id})",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    return _to_response(document)


@router.post("/upload-batch", response_model=BatchUploadResponse, status_code=status.HTTP_200_OK)
async def upload_documents_batch(
    collection_id: str,
    files: list[UploadFile],
    tags: str | None = None,
    user: User = Depends(get_current_user),
    ingest: IngestDocumentUseCase = Depends(get_ingest_use_case),
):
    """Uploads and indexes multiple research papers / files simultaneously in one request."""
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else ["research-paper"]
    results = []
    success_count = 0
    fail_count = 0
    max_bytes = settings.max_upload_mb * 1024 * 1024

    for file in files:
        safe_filename = file.filename or "uploaded_file"
        ext = safe_filename.lower().rsplit(".", 1)[-1] if "." in safe_filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            results.append(
                BatchUploadItemResult(
                    filename=safe_filename,
                    status="failed",
                    error=f"Unsupported file type .{ext}",
                )
            )
            fail_count += 1
            continue

        raw_bytes = await file.read()
        if len(raw_bytes) > max_bytes:
            results.append(
                BatchUploadItemResult(
                    filename=safe_filename,
                    status="failed",
                    error=f"Exceeds {settings.max_upload_mb}MB limit",
                )
            )
            fail_count += 1
            continue

        try:
            doc = await ingest.execute(
                user_id=user.id,
                collection_id=collection_id,
                filename=safe_filename,
                raw_bytes=raw_bytes,
                tags=tag_list,
            )
            chunk_count = doc.metadata.get("chunk_count", 0)
            results.append(
                BatchUploadItemResult(
                    filename=safe_filename,
                    status="indexed",
                    document_id=doc.id,
                    chunk_count=chunk_count,
                )
            )
            success_count += 1
        except DuplicateDocumentError as exc:
            results.append(
                BatchUploadItemResult(
                    filename=safe_filename,
                    status="duplicate",
                    document_id=exc.existing_document_id,
                    error="Already indexed in knowledge base",
                )
            )
        except Exception as exc:
            results.append(
                BatchUploadItemResult(
                    filename=safe_filename,
                    status="failed",
                    error=str(exc),
                )
            )
            fail_count += 1

    return BatchUploadResponse(
        total=len(files),
        successful=success_count,
        failed=fail_count,
        results=results,
    )


@router.post("/import-arxiv", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def import_arxiv_paper(
    payload: ArxivImportRequest,
    user: User = Depends(get_current_user),
    ingest: IngestDocumentUseCase = Depends(get_ingest_use_case),
):
    """Fetches an arXiv research paper by ID/URL, extracts metadata, downloads the PDF, and indexes it."""
    import re
    import xml.etree.ElementTree as ET
    import httpx

    raw_input = payload.arxiv_id_or_url.strip()
    match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", raw_input)
    if not match:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Invalid arXiv ID or URL. Example: 2312.10997 or https://arxiv.org/abs/2312.10997",
        )
    arxiv_id = match.group(1)

    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    headers = {"User-Agent": "Enterprise-RAG-Assistant/1.0 (academic research client)"}
    async with httpx.AsyncClient(timeout=45.0, headers=headers, follow_redirects=True) as client:
        title = f"arXiv_{arxiv_id}"
        try:
            meta_resp = await client.get(api_url)
            if meta_resp.status_code == 200:
                root = ET.fromstring(meta_resp.text)
                entry = root.find("{http://www.w3.org/2005/Atom}entry")
                if entry is not None:
                    title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                    if title_elem is not None and title_elem.text:
                        clean_title = " ".join(title_elem.text.strip().split())
                        # clean any characters that are invalid in filenames
                        clean_title = re.sub(r'[\\/*?:"<>|]', "", clean_title)
                        title = clean_title[:120]
        except Exception:
            pass

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        try:
            pdf_resp = await client.get(pdf_url)
            pdf_resp.raise_for_status()
            pdf_bytes = pdf_resp.content
        except Exception as e:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                f"Failed to download PDF from arXiv: {e}",
            )

    safe_filename = f"{title}.pdf"
    tags = list(set(payload.tags + ["arxiv", f"arxiv:{arxiv_id}"]))

    try:
        document = await ingest.execute(
            user_id=user.id,
            collection_id=payload.collection_id,
            filename=safe_filename,
            raw_bytes=pdf_bytes,
            tags=tags,
        )
    except DuplicateDocumentError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Paper '{safe_filename}' was already uploaded (document_id={exc.existing_document_id})",
        ) from exc
    except Exception as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    return _to_response(document)


@router.post("/upload-async", status_code=status.HTTP_202_ACCEPTED)
async def upload_document_async(
    collection_id: str,
    file: UploadFile,
    user: User = Depends(get_current_user),
):
    """Enqueues ingestion on the Celery task queue instead of processing inline.

    Use this for large files or when you don't want the upload request to
    block on embedding-API latency. Poll GET /documents or GET /documents/{id}
    to see the document transition PENDING -> PROCESSING -> INDEXED|FAILED.
    """
    from knowledge_assistant.infrastructure.tasks.ingestion_tasks import ingest_document_task

    safe_filename = file.filename or "uploaded_file"
    ext = safe_filename.lower().rsplit(".", 1)[-1] if "." in safe_filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported file type: .{ext}")

    raw_bytes = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {settings.max_upload_mb}MB limit")

    async_result = ingest_document_task.delay(user.id, collection_id, safe_filename, raw_bytes)
    return {"task_id": async_result.id, "status": "queued", "filename": safe_filename}



@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    collection_id: str | None = None,
    user: User = Depends(get_current_user),
    repo: SqlDocumentRepository = Depends(get_document_repo),
):
    documents = await repo.list_for_user(user.id, collection_id)
    return [_to_response(d) for d in documents]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    user: User = Depends(get_current_user),
    repo: SqlDocumentRepository = Depends(get_document_repo),
):
    document = await repo.get(document_id, user.id)  # user_id filter enforces isolation
    if not document:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return _to_response(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    user: User = Depends(get_current_user),
    ingest: IngestDocumentUseCase = Depends(get_ingest_use_case),
    repo: SqlDocumentRepository = Depends(get_document_repo),
):
    document = await repo.get(document_id, user.id)
    if not document:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    await ingest.delete_document(document_id, user.id)


def _to_response(document) -> DocumentResponse:
    return DocumentResponse(
        id=document.id, filename=document.filename, doc_type=document.doc_type.value,
        status=document.status.value, version=document.version, tags=document.tags,
        metadata=document.metadata, created_at=document.created_at, updated_at=document.updated_at,
    )
