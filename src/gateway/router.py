"""API routes for gateway and verification endpoints."""

import hashlib
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple
from fastapi import APIRouter, HTTPException, Request, Response, status
from starlette.datastructures import UploadFile

from src.gateway.buffer import buffer_generator_response
from src.gateway.middleware import (
    apply_header_overrides,
    enforce_server_tier_ceiling,
    parse_security_headers,
)
from src.mitigation.generator import BaseGenerator, get_generator
from src.retrieval.ingest import IngestionPipeline
from src.retrieval.parser import extract_text_from_bytes
from src.retrieval.store import DualIndexStore
from src.schemas.ingestion import IngestionResult, KnowledgeIngestRequest
from src.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from src.schemas.verification import (
    MitigationMode,
    VerificationRequest,
    VerificationResponse,
)
from src.verifier.orchestrator import WaterfallOrchestrator, get_orchestrator

router = APIRouter()

# Global default store, orchestrator, and generator instances
# (can be monkeypatched or injected in tests)
_default_store: Optional[DualIndexStore] = None
_default_orchestrator: Optional[WaterfallOrchestrator] = None
_default_generator: Optional[BaseGenerator] = None


def get_knowledge_store() -> DualIndexStore:
    """Dependency provider for the DualIndexStore instance."""
    global _default_store
    if _default_store is None:
        _default_store = DualIndexStore()
    return _default_store


def get_orchestrator_instance() -> WaterfallOrchestrator:
    """Dependency provider for the WaterfallOrchestrator instance."""
    global _default_orchestrator
    if _default_orchestrator is None:
        store = get_knowledge_store()
        _default_orchestrator = get_orchestrator(store=store)
    return _default_orchestrator


def set_orchestrator_instance(orchestrator: Optional[WaterfallOrchestrator]) -> None:
    """Allow setting or overriding the orchestrator instance (e.g. for testing)."""
    global _default_orchestrator
    _default_orchestrator = orchestrator


def get_generator_instance() -> BaseGenerator:
    """Dependency provider for the local LLM BaseGenerator instance."""
    global _default_generator
    if _default_generator is None:
        _default_generator = get_generator()
    return _default_generator


def set_generator_instance(generator: Optional[BaseGenerator]) -> None:
    """Allow setting or overriding the generator instance (e.g. for testing)."""
    global _default_generator
    _default_generator = generator


def format_chat_prompt(messages: Sequence[ChatMessage]) -> Tuple[str, str]:
    """Extract generation prompt and user query from chat messages.

    Returns:
        Tuple of (generation_prompt, user_query).
    """
    if not messages:
        return "", ""

    # Single user message optimization
    if len(messages) == 1 and messages[0].role.lower() == "user":
        return messages[0].content, messages[0].content

    user_msgs = [m.content for m in messages if m.role.lower() == "user"]
    user_query = user_msgs[-1] if user_msgs else messages[-1].content

    dialogue_lines: List[str] = []
    for msg in messages:
        role = msg.role.strip().capitalize()
        dialogue_lines.append(f"{role}: {msg.content}")

    generation_prompt = "\n".join(dialogue_lines)
    return generation_prompt, user_query


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
async def verify_endpoint(
    request: VerificationRequest, raw_request: Request
) -> VerificationResponse:
    """NLI-aware Waterfall verification sidecar endpoint (REQ-GW-02, REQ-GW-05, REQ-RET-05..06).

    Coordinates deterministic claim segmentation, tiered waterfall retrieval,
    and CPU-based NLI inference to deliver sentence-level reliability metrics.
    """
    effective_request = apply_header_overrides(request, raw_request.headers)
    orchestrator = get_orchestrator_instance()
    return orchestrator.verify(effective_request)


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
    response_model=ChatCompletionResponse,
    tags=["Gateway Proxy"],
    summary="OpenAI-compatible chat completions reverse proxy with full response buffering",
    status_code=status.HTTP_200_OK,
)
async def chat_completions(
    payload: ChatCompletionRequest,
    raw_request: Request,
    response: Response,
) -> ChatCompletionResponse:
    """OpenAI-compatible reverse proxy endpoint with 100% response buffering.

    Implements REQ-GW-01, REQ-GW-03, REQ-GW-05.
    Accepts standard OpenAI chat completion schemas, prompts the local Gemma model,
    buffers 100% of the raw response, runs full dehallucination verification with
    default mitigation_mode='rewrite' (ADR-0001, ADR-0008), injects verification
    metadata headers, and returns a verified OpenAI ChatCompletion response.
    """
    if not payload.messages:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="ChatCompletionRequest 'messages' list must contain at least one message.",
        )

    if payload.stream:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                "Streaming replay (stream: true) requires Sprint 2 Ticket 11 "
                "(Synthetic SSE Replay Engine). Set stream: false or omit stream."
            ),
        )

    overrides = parse_security_headers(raw_request.headers)

    requested_tier = overrides.get(
        "max_retrieval_tier",
        payload.max_retrieval_tier if payload.max_retrieval_tier is not None else 2,
    )
    effective_tier, _ = enforce_server_tier_ceiling(requested_tier)

    # Defaults to mitigation_mode = rewrite for reverse proxy (Ticket 10 AC)
    effective_mitigation_mode = overrides.get(
        "mitigation_mode",
        payload.mitigation_mode if payload.mitigation_mode is not None else MitigationMode.REWRITE,
    )

    gen_prompt, user_query = format_chat_prompt(payload.messages)
    generator = get_generator_instance()

    max_new_tokens = payload.max_tokens if payload.max_tokens is not None else 512
    temperature = payload.temperature if payload.temperature is not None else 0.0

    # Buffer 100% of raw model output prior to verification (REQ-GW-03, CON-04, ADR-0008)
    response_buffer = buffer_generator_response(
        generator=generator,
        prompt=gen_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    raw_text = response_buffer.content

    verification_request = VerificationRequest(
        text=raw_text,
        prompt=user_query,
        context_documents=payload.context_documents or [],
        max_retrieval_tier=effective_tier,
        mitigation_mode=effective_mitigation_mode,
    )

    orchestrator = get_orchestrator_instance()
    verification_response = orchestrator.verify(verification_request)

    response.headers["X-Dehallucination-Safe"] = (
        "true" if verification_response.is_safe else "false"
    )
    response.headers["X-Hallucination-Rate"] = f"{verification_response.hallucination_rate:.2f}"
    response.headers["X-Verification-Rate"] = f"{verification_response.verification_rate:.2f}"
    response.headers["X-Needs-Review"] = "true" if verification_response.needs_review else "false"

    sanitized_output = verification_response.sanitized_text

    prompt_tokens = max(1, len(gen_prompt.split()))
    completion_tokens = max(1, len(sanitized_output.split())) if sanitized_output.strip() else 0

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created_ts = int(time.time())

    return ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=created_ts,
        model=payload.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=sanitized_output,
                ),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
