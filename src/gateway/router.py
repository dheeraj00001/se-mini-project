"""API routes for gateway and verification endpoints."""

import hashlib
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Request, status
from starlette.datastructures import UploadFile

from src.retrieval.ingest import IngestionPipeline
from src.retrieval.parser import extract_text_from_bytes
from src.retrieval.store import DualIndexStore
from src.schemas.ingestion import IngestionResult, KnowledgeIngestRequest
from src.schemas.verification import VerificationRequest, VerificationResponse

router = APIRouter()

# Global default store instance (can be monkeypatched or injected in tests)
_default_store: Optional[DualIndexStore] = None


def get_knowledge_store() -> DualIndexStore:
    """Dependency provider for the DualIndexStore instance."""
    global _default_store
    if _default_store is None:
        _default_store = DualIndexStore()
    return _default_store


@router.get(
    "/health",
    tags=["System"],
    summary="System health check",
    status_code=status.HTTP_200_OK,
)
async def health_check() -> Dict[str, Any]:
    """System health and version check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@router.post(
    "/verify",
    response_model=VerificationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Verification"],
    summary="Verify factual claims against tiered evidence sources",
)
async def verify_endpoint(request: VerificationRequest) -> VerificationResponse:
    """Foundational verification endpoint stub (REQ-GW-02 / UT_GW_02).

    Accepts text, prompt, context_documents, max_retrieval_tier, and mitigation_mode.
    Returns a schema-compliant VerificationResponse.
    """
    return VerificationResponse(
        is_safe=True,
        needs_review=False,
        hallucination_rate=0.0,
        verification_rate=0.0,
        unverified_rate=0.0,
        claims=[],
        sanitized_text=request.text,
    )


@router.post(
    "/knowledge/ingest",
    response_model=IngestionResult,
    status_code=status.HTTP_201_CREATED,
    tags=["Retrieval"],
    summary="Ingest reference documents into Qdrant vector store and BM25 index",
)
async def ingest_endpoint(request: Request) -> IngestionResult:
    """Ingest reference documents (PDF, MD, TXT) via file upload or JSON payload.

    Implements REQ-RET-01, REQ-RET-02, REQ-RET-03, and ADR-0007.
    """
    store = get_knowledge_store()
    pipeline = IngestionPipeline(store=store)

    content_type = request.headers.get("content-type", "").lower()

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_file = form.get("file")
        if not isinstance(uploaded_file, UploadFile):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Multipart request must contain a 'file' field",
            )

        filename = uploaded_file.filename or "uploaded_document"
        raw_bytes = await uploaded_file.read()
        try:
            extracted = extract_text_from_bytes(raw_bytes, filename)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e

        doc_hash = hashlib.sha256(raw_bytes).hexdigest()
        return pipeline.ingest_text(extracted, source_doc=filename, doc_hash=doc_hash)

    else:
        # JSON body ingestion
        try:
            body = await request.json()
            payload = KnowledgeIngestRequest.model_validate(body)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid JSON request body: {e}",
            )

        if payload.source_path:
            from pathlib import Path

            p = Path(payload.source_path)
            if p.is_file():
                return pipeline.ingest_file(p)
            elif p.is_dir():
                return pipeline.ingest_directory(p)
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Source path not found: {payload.source_path}",
                )
        elif payload.text:
            return pipeline.ingest_text(
                payload.text,
                source_doc=payload.source_doc or "api_submission.txt",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="JSON payload must provide either 'text' or 'source_path'",
            )


@router.post(
    "/v1/chat/completions",
    tags=["Gateway Proxy"],
    summary="OpenAI-compatible chat completions proxy stub",
    status_code=status.HTTP_200_OK,
)
async def chat_completions_stub(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Foundational OpenAI-compatible reverse proxy stub (REQ-GW-01 / UT_GW_01)."""
    return {
        "id": "chatcmpl-stub-001",
        "object": "chat.completion",
        "model": payload.get("model", "gemma-4b"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Dehallucination proxy initialized.",
                },
                "finish_reason": "stop",
            }
        ],
    }
