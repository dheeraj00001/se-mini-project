"""Targeted mitigation, local generator, and secondary verification subsystem.

Implements REQ-MIT-01, REQ-MIT-02, REQ-MIT-03, REQ-MIT-04, REQ-MIT-05,
NFR-SAFE-01, NFR-SAFE-02, ADR-0002, ADR-0006.
"""

from src.mitigation.fallback import (
    DEFAULT_REDACTION_TOKEN,
    DEFAULT_REFUSAL_MESSAGE,
    FallbackEngine,
    MitigationBlockedError,
    redact_contradicted_claims,
)
from src.mitigation.generator import (
    BaseGenerator,
    MockGenerator,
    QuantizedGemmaGenerator,
    get_generator,
)
from src.mitigation.pipeline import (
    MitigationPipeline,
    get_mitigation_pipeline,
)
from src.mitigation.rewriter import (
    GemmaRewriter,
    TargetedRewriter,
    build_rewrite_prompt,
    construct_targeted_rewrite_prompt,
    filter_contradicted_claims,
)

__all__ = [
    "BaseGenerator",
    "MockGenerator",
    "QuantizedGemmaGenerator",
    "get_generator",
    "TargetedRewriter",
    "GemmaRewriter",
    "build_rewrite_prompt",
    "construct_targeted_rewrite_prompt",
    "filter_contradicted_claims",
    "DEFAULT_REDACTION_TOKEN",
    "DEFAULT_REFUSAL_MESSAGE",
    "FallbackEngine",
    "MitigationBlockedError",
    "redact_contradicted_claims",
    "MitigationPipeline",
    "get_mitigation_pipeline",
]
