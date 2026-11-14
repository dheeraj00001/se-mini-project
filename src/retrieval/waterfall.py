"""Waterfall retrieval orchestration module (REQ-RET-05, ADR-0004).

Provides re-exports for WaterfallOrchestrator and helper factory to satisfy
SRS traceability (REQ-RET-05, Section 5.3, Table 8.2).
"""

from src.verifier.orchestrator import (
    VerificationOrchestrator,
    WaterfallOrchestrator,
    get_orchestrator,
)

__all__ = [
    "VerificationOrchestrator",
    "WaterfallOrchestrator",
    "get_orchestrator",
]
