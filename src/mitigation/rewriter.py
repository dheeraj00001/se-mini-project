"""Targeted Prompt Rewriter for Contradicted Factual Claims.

Implements REQ-MIT-02, REQ-MIT-05, ADR-0006, BR-01, BR-02.

Core Responsibilities:
1. filter_contradicted_claims: Surgically isolates only CONTRADICTED claims (P >= 0.70).
2. Enforces Business Rule BR-01: UNVERIFIED claims are never modified, erased, or rewritten.
3. build_rewrite_prompt: Assembles structured instruction prompts pairing original text,
   contradicted claims, and grounding evidence premises.
4. TargetedRewriter (and alias GemmaRewriter): Orchestrates the generator to produce
   corrected text replacing factual falsehoods while strictly preserving supported and
   unverified propositions.
"""

import re
from typing import List, Optional, Sequence
from src.mitigation.generator import BaseGenerator, get_generator
from src.schemas.verification import ClaimReport, ClaimState


def filter_contradicted_claims(
    claims: Sequence[ClaimReport],
    contradiction_threshold: float = 0.70,
) -> List[ClaimReport]:
    """Isolate claims eligible for targeted rewriting (REQ-MIT-02, BR-01, BR-02).

    Strictly applies Business Rule BR-01:
    - Claims with state CONTRADICTED and P(contradiction) >= contradiction_threshold are selected.
    - Claims with state SUPPORTED are excluded.
    - Claims with state UNVERIFIED are strictly excluded.

    Args:
        claims: Sequence of ClaimReport evaluations.
        contradiction_threshold: Minimum contradiction probability (default 0.70).

    Returns:
        List of ClaimReport objects meeting the contradiction criteria.
    """
    eligible: List[ClaimReport] = []
    for claim in claims:
        # Enforce BR-01: unverified is never treated as contradiction
        if claim.state != ClaimState.CONTRADICTED:
            continue

        p_contra = claim.probabilities.get("contradiction", 0.0)
        if p_contra >= contradiction_threshold:
            eligible.append(claim)

    return eligible


def build_rewrite_prompt(
    original_text: str,
    contradicted_claims: Sequence[ClaimReport],
) -> str:
    """Construct structured targeted rewrite prompt for the local Gemma generator.

    Implements REQ-MIT-02, REQ-MIT-05, ADR-0006.

    Args:
        original_text: The source generated response requiring correction.
        contradicted_claims: Contradicted ClaimReport items paired with evidence premises.

    Returns:
        Formatted prompt string directing surgical factual replacement.
    """
    if not contradicted_claims:
        return ""

    items_block: List[str] = []
    for idx, claim in enumerate(contradicted_claims, start=1):
        premise = (claim.evidence_text or "").strip()
        if not premise:
            premise = "Evidence confirms the statement is factually contradicted."
        items_block.append(
            f"CONTRADICTED CLAIM {idx}: {claim.claim_text}\n" f"EVIDENCE PREMISE: {premise}"
        )

    claims_text = "\n\n".join(items_block)

    return (
        "You are a precise factual rewriting assistant. Your task is to update ONLY the "
        "contradicted claims in the original text using the verified evidence premises "
        "provided.\n\n"
        "STRICT GUIDELINES:\n"
        "1. Replace the contradictory assertion with the fact from the corresponding "
        "evidence premise.\n"
        "2. DO NOT modify, omit, or rephrase any statements that are supported or "
        "factually correct.\n"
        "3. DO NOT modify, rephrase, or remove any unverified statements or surrounding context. "
        "Preserve all other content exactly as written.\n"
        "4. Maintain the tone, style, and natural flow of the original text.\n"
        "5. Output ONLY the finalized rewritten text without conversational filler, "
        "markdown code fences, or explanations.\n\n"
        f"ORIGINAL TEXT:\n{original_text.strip()}\n\n"
        f"{claims_text}\n\n"
        "REWRITTEN TEXT:"
    )


# Domain alias
construct_targeted_rewrite_prompt = build_rewrite_prompt


class TargetedRewriter:
    """Targeted Rewriter using local Gemma generator for surgical hallucination correction.

    Implements REQ-MIT-01, REQ-MIT-02, REQ-MIT-05, ADR-0006.
    """

    def __init__(
        self,
        generator: Optional[BaseGenerator] = None,
        contradiction_threshold: float = 0.70,
    ) -> None:
        self._generator = generator if generator is not None else get_generator()
        self._contradiction_threshold = contradiction_threshold

    @property
    def generator(self) -> BaseGenerator:
        """The underlying text generator instance."""
        return self._generator

    @property
    def contradiction_threshold(self) -> float:
        """Threshold above which a claim is eligible for rewrite."""
        return self._contradiction_threshold

    def rewrite(
        self,
        original_text: str,
        claims: Optional[Sequence[ClaimReport]] = None,
        *,
        contradicted_claims: Optional[Sequence[ClaimReport]] = None,
    ) -> str:
        """Execute prompt-guided targeted rewrite on text containing contradicted claims.

        Args:
            original_text: The original response string.
            claims: Evaluation reports for claims in original_text (can contain
                    supported, unverified, and contradicted claims).
            contradicted_claims: Optional keyword argument alias for claims (DoD compatibility).

        Returns:
            Sanitized and factually corrected text string.
        """
        if not original_text or not original_text.strip():
            return original_text

        candidate_claims: Sequence[ClaimReport] = []
        if claims is not None:
            candidate_claims = claims
        elif contradicted_claims is not None:
            candidate_claims = contradicted_claims

        # 1. Filter strictly for contradicted claims above threshold (BR-01, BR-02)
        contradicted = filter_contradicted_claims(
            candidate_claims,
            contradiction_threshold=self._contradiction_threshold,
        )

        # 2. Short-circuit: if no contradicted claims exist, return original text unchanged
        if not contradicted:
            return original_text

        # 3. Build targeted prompt
        prompt = build_rewrite_prompt(
            original_text=original_text,
            contradicted_claims=contradicted,
        )

        # 4. Invoke local generator
        raw_output = self._generator.generate(
            prompt=prompt,
            max_new_tokens=len(original_text.split()) + 256,
            temperature=0.0,
        )

        # 5. Clean output
        sanitized = self._clean_generated_output(raw_output)
        return sanitized if sanitized else original_text

    def _clean_generated_output(self, output: str) -> str:
        """Strip prompt artifacts, code fences, or leading headers from generated output."""
        text = output.strip()

        # Remove markdown code blocks if the generator wrapped output in ```markdown or ```
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
                text = "\n".join(lines[1:-1]).strip()

        # Remove "REWRITTEN TEXT:" prefix if echoed
        text = re.sub(r"^REWRITTEN TEXT:\s*", "", text, flags=re.IGNORECASE)

        return text.strip()


# Definition of Done Alias
GemmaRewriter = TargetedRewriter
