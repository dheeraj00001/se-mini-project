"""Data schemas for inter-module communication."""

from src.schemas.ingestion import IngestionResult, KnowledgeIngestRequest
from src.schemas.verification import (
    AtomicClaim,
    ClaimReport,
    ClaimSegment,
    ClaimState,
    EvidenceChunk,
    MitigationMode,
    VerificationRequest,
    VerificationResponse,
)

__all__ = [
    "AtomicClaim",
    "ClaimReport",
    "ClaimSegment",
    "ClaimState",
    "EvidenceChunk",
    "IngestionResult",
    "KnowledgeIngestRequest",
    "MitigationMode",
    "VerificationRequest",
    "VerificationResponse",
]
