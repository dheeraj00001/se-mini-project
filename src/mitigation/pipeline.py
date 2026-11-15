"""Secondary Verification Loop and Mitigation Pipeline Integration.

Implements REQ-MIT-03, REQ-MIT-04, NFR-SAFE-01, NFR-SAFE-02, ADR-0006, BR-01, BR-02.

Core Responsibilities:
1. MitigationPipeline: Coordinates targeted rewriting, mandatory single-pass secondary verification,
   and automated fallback cascade.
2. Single Secondary Verification Pass (REQ-MIT-03):
   Re-extracts claims from rewritten candidate text and evaluates them against tiered knowledge.
3. Strict Recursion Prevention (NFR-SAFE-02):
   Limits verification to at most one secondary pass per request; multi-pass loops prohibited.
4. Fallback Cascade (REQ-MIT-04):
   If secondary verification detects residual contradictions (P >= 0.70) or trips the safety
   gate (is_safe == False), immediately terminates the correction loop and cascades to REDACT
   (sentence masking) or BLOCK (hard refusal).
"""

from typing import Any, Optional, Sequence
from src.mitigation.fallback import (
    DEFAULT_REDACTION_TOKEN,
    DEFAULT_REFUSAL_MESSAGE,
    FallbackEngine,
    MitigationBlockedError,
)
from src.mitigation.rewriter import (
    TargetedRewriter,
    filter_contradicted_claims,
)
from src.schemas.verification import (
    ClaimReport,
    MitigationMode,
    VerificationRequest,
    VerificationResponse,
)


class MitigationPipeline:
    """Orchestrates targeted mitigation and single-pass secondary verification guardrails."""

    def __init__(
        self,
        rewriter: Optional[TargetedRewriter] = None,
        orchestrator: Optional[Any] = None,
        fallback_engine: Optional[FallbackEngine] = None,
        fallback_mode: MitigationMode = MitigationMode.REDACT,
        contradiction_threshold: float = 0.70,
        severe_contradiction_threshold: float = 0.85,
    ) -> None:
        self._rewriter = rewriter if rewriter is not None else TargetedRewriter()
        self._orchestrator = orchestrator
        self._fallback_engine = (
            fallback_engine
            if fallback_engine is not None
            else FallbackEngine(
                contradiction_threshold=contradiction_threshold,
                redaction_token=DEFAULT_REDACTION_TOKEN,
                refusal_message=DEFAULT_REFUSAL_MESSAGE,
            )
        )
        self._fallback_mode = fallback_mode
        self._contradiction_threshold = contradiction_threshold
        self._severe_contradiction_threshold = severe_contradiction_threshold

    @property
    def rewriter(self) -> TargetedRewriter:
        """The underlying TargetedRewriter instance."""
        return self._rewriter

    @property
    def orchestrator(self) -> Optional[Any]:
        """The verification orchestrator for secondary verification passes."""
        return self._orchestrator

    @property
    def fallback_engine(self) -> FallbackEngine:
        """The fallback engine managing redaction and blocking."""
        return self._fallback_engine

    @property
    def fallback_mode(self) -> MitigationMode:
        """Configured default fallback strategy (REDACT or BLOCK)."""
        return self._fallback_mode

    def execute(
        self,
        request: VerificationRequest,
        initial_response: VerificationResponse,
        raise_on_block: bool = True,
    ) -> VerificationResponse:
        """Execute mitigation strategy and secondary verification on initial response.

        Args:
            request: Client verification request containing configuration and modes.
            initial_response: Initial VerificationResponse containing segmented claims and rates.
            raise_on_block: Whether to raise MitigationBlockedError (HTTP 422) if blocked.

        Returns:
            VerificationResponse with approved, rewritten, or redacted sanitized_text.
        """
        mode = request.mitigation_mode

        # -------------------------------------------------------------
        # Mode 1: FLAG - Passive reporting (default for /verify)
        # -------------------------------------------------------------
        if mode == MitigationMode.FLAG:
            return initial_response.model_copy(update={"sanitized_text": request.text})

        contradicted_claims = filter_contradicted_claims(
            initial_response.claims,
            contradiction_threshold=self._contradiction_threshold,
        )

        # -------------------------------------------------------------
        # Mode 2: REDACT - Direct sentence masking
        # -------------------------------------------------------------
        if mode == MitigationMode.REDACT:
            if not contradicted_claims:
                return initial_response.model_copy(update={"sanitized_text": request.text})
            redacted = self._fallback_engine.redact(request.text, initial_response.claims)
            return initial_response.model_copy(update={"sanitized_text": redacted})

        # -------------------------------------------------------------
        # Mode 3: BLOCK - Hard refusal circuit breaker (NFR-SAFE-01)
        # -------------------------------------------------------------
        if mode == MitigationMode.BLOCK:
            if not contradicted_claims and initial_response.is_safe:
                return initial_response.model_copy(update={"sanitized_text": request.text})
            return self._handle_block_action(
                prior_response=initial_response,
                raise_on_block=raise_on_block,
            )

        # -------------------------------------------------------------
        # Mode 4: REWRITE - Targeted Rewrite + Secondary Verification
        # -------------------------------------------------------------
        effective_fallback = request.fallback_mode or self._fallback_mode

        # If zero contradictions and safety gate is satisfied, return original text unchanged
        if not contradicted_claims and initial_response.is_safe:
            return initial_response.model_copy(update={"sanitized_text": request.text})

        # 1. Execute targeted factual rewrite using local Gemma generator (REQ-MIT-02)
        rewritten_text = self._rewriter.rewrite(
            request.text,
            claims=initial_response.claims,
        )

        # If rewriting failed to modify text or returned blank, cascade immediately
        if not rewritten_text.strip() or rewritten_text.strip() == request.text.strip():
            return self._handle_fallback_cascade(
                mode=effective_fallback,
                candidate_text=request.text,
                claims=initial_response.claims,
                prior_response=initial_response,
                raise_on_block=raise_on_block,
            )

        # 2. Mandatory single secondary verification pass (REQ-MIT-03, NFR-SAFE-02)
        secondary_response = self._run_secondary_verification(
            rewritten_text=rewritten_text,
            original_request=request,
        )

        # 3. Evaluate secondary verification results
        secondary_contradicted = filter_contradicted_claims(
            secondary_response.claims,
            contradiction_threshold=self._contradiction_threshold,
        )

        # Clean verdict: Zero residual contradictions and is_safe == True -> Approved!
        if not secondary_contradicted and secondary_response.is_safe:
            return secondary_response.model_copy(update={"sanitized_text": rewritten_text})

        # 4. Secondary verification detected residual contradictions OR tripped safety gate
        # Recursion Prevention (NFR-SAFE-02): Stop immediately and cascade (REQ-MIT-04)
        return self._handle_fallback_cascade(
            mode=effective_fallback,
            candidate_text=rewritten_text,
            claims=secondary_response.claims,
            prior_response=secondary_response,
            raise_on_block=raise_on_block,
        )

    def _run_secondary_verification(
        self,
        rewritten_text: str,
        original_request: VerificationRequest,
    ) -> VerificationResponse:
        """Execute a single secondary verification pass on rewritten candidate text (REQ-MIT-03)."""
        orchestrator = self._orchestrator
        if orchestrator is None:
            from src.verifier.orchestrator import get_orchestrator

            orchestrator = get_orchestrator()
            self._orchestrator = orchestrator

        secondary_request = VerificationRequest(
            text=rewritten_text,
            prompt=original_request.prompt,
            context_documents=original_request.context_documents,
            max_retrieval_tier=original_request.max_retrieval_tier,
            mitigation_mode=MitigationMode.FLAG,  # Enforces NFR-SAFE-02: never recurse
        )

        # Execute single secondary pass on orchestrator
        return orchestrator.verify(secondary_request, _is_secondary=True)

    def _handle_block_action(
        self,
        prior_response: VerificationResponse,
        raise_on_block: bool,
    ) -> VerificationResponse:
        """Handle BLOCK action by raising MitigationBlockedError or returning refusal text."""
        if raise_on_block:
            raise MitigationBlockedError(
                detail=self._fallback_engine.refusal_message,
                response=prior_response,
            )
        return prior_response.model_copy(
            update={
                "sanitized_text": self._fallback_engine.refusal_message,
                "is_safe": False,
            }
        )

    def _handle_fallback_cascade(
        self,
        mode: MitigationMode,
        candidate_text: str,
        claims: Sequence[ClaimReport],
        prior_response: VerificationResponse,
        raise_on_block: bool,
    ) -> VerificationResponse:
        """Cascade to configured fallback action (redact or block) upon secondary failure."""
        if mode == MitigationMode.BLOCK:
            return self._handle_block_action(
                prior_response=prior_response,
                raise_on_block=raise_on_block,
            )

        # Default fallback: REDACT
        redacted = self._fallback_engine.redact(candidate_text, claims)
        return prior_response.model_copy(
            update={
                "sanitized_text": redacted,
            }
        )


def get_mitigation_pipeline(
    rewriter: Optional[TargetedRewriter] = None,
    orchestrator: Optional[Any] = None,
    fallback_engine: Optional[FallbackEngine] = None,
    fallback_mode: MitigationMode = MitigationMode.REDACT,
    **kwargs: Any,
) -> MitigationPipeline:
    """Factory creating an instantiated MitigationPipeline."""
    return MitigationPipeline(
        rewriter=rewriter,
        orchestrator=orchestrator,
        fallback_engine=fallback_engine,
        fallback_mode=fallback_mode,
        **kwargs,
    )
