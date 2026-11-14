"""API routes for gateway and verification endpoints."""

from typing import Any, Dict
from fastapi import APIRouter, status
from src.schemas.verification import VerificationRequest, VerificationResponse

router = APIRouter()


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
