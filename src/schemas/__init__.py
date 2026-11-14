"""Data schemas for inter-module communication."""

from src.schemas.ingestion import IngestionResult, KnowledgeIngestRequest
from src.schemas.verification import (
    AtomicClaim,
    ClaimReport,
    ClaimSegment,
    ClaimState,
    DEBERTA_NLI_LABEL_ORDER,
    DEBERTA_TO_CLAIM_STATE,
    EvidenceChunk,
    MitigationMode,
    VerificationRequest,
    VerificationResponse,
    determine_claim_state_from_nli,
    map_nli_probabilities,
)

__all__ = [
    "AtomicClaim",
    "ClaimReport",
    "ClaimSegment",
    "ClaimState",
    "DEBERTA_NLI_LABEL_ORDER",
    "DEBERTA_TO_CLAIM_STATE",
    "EvidenceChunk",
    "IngestionResult",
    "KnowledgeIngestRequest",
    "MitigationMode",
    "VerificationRequest",
    "VerificationResponse",
    "determine_claim_state_from_nli",
    "map_nli_probabilities",
]
