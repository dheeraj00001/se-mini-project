"""Data schemas for inter-module communication."""

from src.schemas.verification import (
    ClaimReport,
    ClaimSegment,
    ClaimState,
    EvidenceChunk,
    MitigationMode,
    VerificationRequest,
    VerificationResponse,
)

__all__ = [
    "ClaimReport",
    "ClaimSegment",
    "ClaimState",
    "EvidenceChunk",
    "MitigationMode",
    "VerificationRequest",
    "VerificationResponse",
]
