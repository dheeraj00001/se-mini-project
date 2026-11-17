"""Verifier module containing claim segmentation and NLI verification."""

from src.verifier.nli import (
    BaseNLIVerifier,
    DebertaNLIVerifier,
    MockNLIVerifier,
    NLIVerifier,
    calibrate_safety_circuit_breaker,
    compute_document_rates,
    compute_softmax,
    determine_claim_state,
    evaluate_safety_gate,
    get_nli_verifier,
    map_deberta_probabilities,
)
from src.verifier.orchestrator import (
    WaterfallOrchestrator,
    VerificationOrchestrator,
    get_orchestrator,
    get_verification_telemetry_summary,
)
from src.verifier.segmenter import ClaimSegmenter

__all__ = [
    "BaseNLIVerifier",
    "ClaimSegmenter",
    "DebertaNLIVerifier",
    "MockNLIVerifier",
    "NLIVerifier",
    "VerificationOrchestrator",
    "WaterfallOrchestrator",
    "calibrate_safety_circuit_breaker",
    "compute_document_rates",
    "compute_softmax",
    "determine_claim_state",
    "evaluate_safety_gate",
    "get_nli_verifier",
    "get_orchestrator",
    "get_verification_telemetry_summary",
    "map_deberta_probabilities",
]
