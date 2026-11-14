"""API routes for gateway and verification endpoints."""

from fastapi import APIRouter, status
from src.schemas.verification import VerificationRequest, VerificationResponse

router = APIRouter()


@router.post(
    "/verify",
    response_model=VerificationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Verification"],
    summary="Verify factual claims against tiered evidence sources",
)
async def verify_endpoint(request: VerificationRequest) -> VerificationResponse:
    """Foundational verification endpoint stub (REQ-GW-02).

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
