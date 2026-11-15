"""Mitigation Fallback Cascade Engine.

Implements REQ-MIT-04, NFR-SAFE-01, ADR-0006, BR-01, BR-02.

Core Responsibilities:
1. redact: Surgically replaces contradicted statements with standardized redaction notice:
   "[Claim redacted: Disproven by reference knowledge]"
   while strictly preserving supported and unverified propositions (BR-01).
2. block: Emits structured hard refusal response (HTTP 422 MitigationBlockedError or canonical
   refusal text: "Unable to provide a factually grounded response.") preventing release of
   ungrounded responses (NFR-SAFE-01).
3. FallbackEngine: Unified coordinator dispatching fallback mitigation strategies.
"""

from typing import List, Optional, Sequence
from fastapi import HTTPException
import nltk

from src.mitigation.rewriter import filter_contradicted_claims
from src.schemas.verification import (
    ClaimReport,
    MitigationMode,
    VerificationResponse,
)
from src.verifier.segmenter import _ensure_nltk_resources

DEFAULT_REDACTION_TOKEN: str = "[Claim redacted: Disproven by reference knowledge]"
DEFAULT_REFUSAL_MESSAGE: str = "Unable to provide a factually grounded response."


class MitigationBlockedError(HTTPException):
    """Raised when mitigation mode is BLOCK and contradictions exist (REQ-MIT-04, NFR-SAFE-01).

    Inherits from HTTPException with status 422 to seamlessly integrate with FastAPI error routing.
    """

    def __init__(
        self,
        detail: str = DEFAULT_REFUSAL_MESSAGE,
        response: Optional[VerificationResponse] = None,
    ) -> None:
        super().__init__(
            status_code=422,
            detail=detail,
        )
        self.response = response


def redact_contradicted_claims(
    text: str,
    claims: Sequence[ClaimReport],
    contradiction_threshold: float = 0.70,
    redaction_token: str = DEFAULT_REDACTION_TOKEN,
) -> str:
    """Surgically replace contradicted statements with standardized redaction notice.

    Implements REQ-MIT-04, BR-01.

    Args:
        text: Source text containing potential contradicted statements.
        claims: Evaluated claims for the text.
        contradiction_threshold: Minimum contradiction probability (default 0.70).
        redaction_token: Standardized notice replacing contradicted sentences.

    Returns:
        Sanitized text with contradicted sentences replaced by redaction_token.
    """
    if not text or not text.strip():
        return text

    # Filter strictly for contradicted claims above threshold (BR-01, BR-02)
    contradicted = filter_contradicted_claims(
        claims,
        contradiction_threshold=contradiction_threshold,
    )
    if not contradicted:
        return text

    _ensure_nltk_resources()
    sentences = nltk.sent_tokenize(text)

    # Identify sentences matching contradicted claims
    matched_sentences: List[str] = []
    for sent in sentences:
        sent_clean = sent.strip()
        is_contra = False
        for c in contradicted:
            orig = (getattr(c, "original_sentence", None) or "").strip()
            if orig and (orig == sent_clean or orig in sent or sent_clean in orig):
                is_contra = True
                break
            claim_text = (c.claim_text or "").strip()
            if claim_text and claim_text.lower() in sent_clean.lower():
                is_contra = True
                break
        if is_contra:
            matched_sentences.append(sent)

    redacted = text
    for sent in matched_sentences:
        if sent in redacted:
            redacted = redacted.replace(sent, redaction_token)

    return redacted


class FallbackEngine:
    """Coordinates automated fallback mitigation strategies (REQ-MIT-04, NFR-SAFE-01)."""

    def __init__(
        self,
        contradiction_threshold: float = 0.70,
        redaction_token: str = DEFAULT_REDACTION_TOKEN,
        refusal_message: str = DEFAULT_REFUSAL_MESSAGE,
    ) -> None:
        self._contradiction_threshold = contradiction_threshold
        self._redaction_token = redaction_token
        self._refusal_message = refusal_message

    @property
    def contradiction_threshold(self) -> float:
        """Threshold above which a claim is considered contradicted."""
        return self._contradiction_threshold

    @property
    def redaction_token(self) -> str:
        """Standardized token for sentence redaction."""
        return self._redaction_token

    @property
    def refusal_message(self) -> str:
        """Canonical refusal string for blocked responses."""
        return self._refusal_message

    def redact(
        self,
        text: str,
        claims: Sequence[ClaimReport],
        redaction_token: Optional[str] = None,
    ) -> str:
        """Surgically replace contradicted statements with standardized redaction tokens."""
        token = redaction_token if redaction_token is not None else self._redaction_token
        return redact_contradicted_claims(
            text=text,
            claims=claims,
            contradiction_threshold=self._contradiction_threshold,
            redaction_token=token,
        )

    def block(
        self,
        response: Optional[VerificationResponse] = None,
        raise_exception: bool = True,
    ) -> str:
        """Execute hard refusal action preventing release of ungrounded responses (NFR-SAFE-01).

        Args:
            response: Optional underlying VerificationResponse for context.
            raise_exception: If True, raises MitigationBlockedError (HTTP 422).
                             If False, returns canonical refusal string.

        Returns:
            Canonical refusal text when raise_exception is False.
        """
        if raise_exception:
            raise MitigationBlockedError(
                detail=self._refusal_message,
                response=response,
            )
        return self._refusal_message

    def execute_fallback(
        self,
        mode: MitigationMode,
        text: str,
        claims: Sequence[ClaimReport],
        response: Optional[VerificationResponse] = None,
        raise_on_block: bool = True,
    ) -> str:
        """Dispatch fallback action based on configured MitigationMode.

        Args:
            mode: Mitigation mode (REDACT or BLOCK).
            text: Candidate text to redact if mode is REDACT.
            claims: Evaluation reports to identify contradicted statements.
            response: Optional response object for BLOCK error context.
            raise_on_block: Whether to raise MitigationBlockedError when mode is BLOCK.

        Returns:
            Redacted or refusal text string.
        """
        if mode == MitigationMode.BLOCK:
            return self.block(response=response, raise_exception=raise_on_block)
        return self.redact(text=text, claims=claims)
