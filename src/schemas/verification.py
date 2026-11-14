"""Pydantic schemas for the verification pipeline and inter-module contracts."""

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence
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


class ClaimSegment(BaseModel):
    """Atomic factual proposition extracted from text (Module 2 / Student 2 contract)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: int = Field(..., ge=0, description="Sequential index of the atomic claim.")
    claim_text: str = Field(
        ..., min_length=1, description="Standalone atomic hypothesis proposition."
    )
    original_sentence: str = Field(
        ..., min_length=1, description="Source sentence containing the claim."
    )
    has_epistemic_hedge: bool = Field(
        default=False,
        description="True if extracted from under an epistemic hedge (e.g. 'I think').",
    )


# Domain alias as defined in CONTEXT.md and Ticket 02
AtomicClaim = ClaimSegment


class EvidenceChunk(BaseModel):
    """Retrieved grounding knowledge chunk (Module 3 / Student 3 contract)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(..., min_length=1, description="Deterministic SHA-256 hash ID.")
    text: str = Field(..., min_length=1, description="Chunk text passage.")
    source_doc: str = Field(..., min_length=1, description="Source document identifier or path.")
    tier: int = Field(..., ge=1, le=3, description="Grounding tier (1=Context, 2=Local KB, 3=Web).")
    score: Optional[float] = Field(
        default=None,
        description="Reciprocal rank fusion or similarity score.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Passage or document metadata.",
    )


# Domain alias as defined in SRS and Ticket 04
EvidenceCandidate = EvidenceChunk


class ClaimReport(BaseModel):
    """Individual claim evaluation report (SRS Appendix B.3)."""

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
    """Payload schema for POST /verify endpoint (SRS Appendix B.1)."""

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
        return v


class VerificationResponse(BaseModel):
    """Response schema for verification endpoints (SRS Appendix B.2)."""

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
        ...,
        description="Detailed list of individual claim evaluations.",
    )
    sanitized_text: str = Field(
        ...,
        description="Finalized output text (original, rewritten, or redacted).",
    )


# ---------------------------------------------------------------------------
# NLI Model Specifications & Label Mapping (cross-encoder/nli-deberta-v3-base)
# ---------------------------------------------------------------------------
# Hugging Face official index ordering: 0=contradiction, 1=entailment, 2=neutral
DEBERTA_NLI_LABEL_ORDER: tuple[str, ...] = ("contradiction", "entailment", "neutral")

DEBERTA_TO_CLAIM_STATE: Dict[str, ClaimState] = {
    "contradiction": ClaimState.CONTRADICTED,
    "entailment": ClaimState.SUPPORTED,
    "neutral": ClaimState.UNVERIFIED,
}


def map_nli_probabilities(
    probs: Sequence[float],
    label_order: Sequence[str] = DEBERTA_NLI_LABEL_ORDER,
) -> Dict[str, float]:
    """Map raw NLI model probability outputs to explicit label names.

    Ensures that probabilities are mapped by their documented label names
    rather than assuming an arbitrary index ordering.
    """
    if len(probs) != len(label_order):
        raise ValueError(
            f"Expected {len(label_order)} probabilities matching {label_order}, got {len(probs)}"
        )
    return {label: float(prob) for label, prob in zip(label_order, probs)}


def determine_claim_state_from_nli(
    probabilities: Dict[str, float],
    threshold: float = 0.5,
) -> ClaimState:
    """Determine discrete ClaimState using verified label probabilities.

    Args:
        probabilities: Dict containing 'entailment', 'contradiction', and 'neutral'.
        threshold: Minimum confidence required for decisive classification.
    """
    entailment = probabilities.get("entailment", 0.0)
    contradiction = probabilities.get("contradiction", 0.0)

    if contradiction >= threshold and contradiction > entailment:
        return ClaimState.CONTRADICTED
    if entailment >= threshold and entailment > contradiction:
        return ClaimState.SUPPORTED
    return ClaimState.UNVERIFIED
