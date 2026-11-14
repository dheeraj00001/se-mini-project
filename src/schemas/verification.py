"""Pydantic schemas for the verification pipeline."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClaimState(str, Enum):
    """Discrete verification state for an atomic claim."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"


class MitigationMode(str, Enum):
    """Mitigation action upon detecting contradicted claims."""

    FLAG = "flag"
    REWRITE = "rewrite"
    REDACT = "redact"
    BLOCK = "block"


class ClaimReport(BaseModel):
    """Individual claim evaluation report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: int = Field(..., description="Sequence index of the claim.")
    claim_text: str = Field(..., min_length=1, description="Extracted atomic claim text.")
    state: ClaimState = Field(..., description="Discrete verification label.")
    probabilities: Dict[str, float] = Field(
        ...,
        description="NLI probabilities for entailment, neutral, and contradiction.",
    )
    evidence_tier: Optional[int] = Field(
        default=None,
        ge=1,
        le=3,
        description="Grounding tier (1=Context, 2=Local KB, 3=Web).",
    )
    evidence_text: Optional[str] = Field(
        default=None,
        description="Premise passage used for verification.",
    )


class VerificationRequest(BaseModel):
    """Payload schema for POST /verify endpoint."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="Generated AI text to be verified.")
    prompt: str = Field(default="", description="Original user prompt.")
    context_documents: List[str] = Field(
        default_factory=list,
        description="Tier 1 in-context reference passages.",
    )
    max_retrieval_tier: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Maximum search tier allowed (1, 2, or 3).",
    )
    mitigation_mode: MitigationMode = Field(
        default=MitigationMode.FLAG,
        description="Desired action upon detecting contradictions.",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must not be empty or whitespace only")
        return v.strip()


class VerificationResponse(BaseModel):
    """Response schema for verification endpoints."""

    model_config = ConfigDict(extra="forbid")

    is_safe: bool = Field(
        ...,
        description="Overall safety verdict (false if severe contradiction detected).",
    )
    needs_review: bool = Field(
        ...,
        description="Advisory flag (true if unverified claims exist).",
    )
    hallucination_rate: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of claims classified as CONTRADICTED.",
    )
    verification_rate: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of claims classified as SUPPORTED.",
    )
    unverified_rate: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of claims classified as UNVERIFIED.",
    )
    claims: List[ClaimReport] = Field(
        default_factory=list,
        description="Detailed list of individual claim evaluations.",
    )
    sanitized_text: str = Field(
        ...,
        description="Finalized output text (original, rewritten, or redacted).",
    )
