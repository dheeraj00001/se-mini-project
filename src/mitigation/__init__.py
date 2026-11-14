"""Targeted mitigation and local generator subsystem.

Implements REQ-MIT-01, REQ-MIT-02, REQ-MIT-05, ADR-0002, ADR-0006.
"""

from src.mitigation.generator import (
    BaseGenerator,
    MockGenerator,
    QuantizedGemmaGenerator,
    get_generator,
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
]
