"""Deterministic sentence and clause claim segmentation engine (REQ-VER-01..03)."""

import re
from typing import List, Optional
import nltk
from src.schemas.verification import AtomicClaim

# Regex patterns for non-factual conversational filler and greetings
_CONVERSATIONAL_FILLER_PATTERNS = [
    re.compile(
        r"^(?:hello|hi|hey|good (?:morning|afternoon|evening)|greetings)\b.*[.!:]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:sure|certainly|of course|absolutely)(?:,\s*|\s+)?"
        r"(?:i can help(?: with that)?|here is the answer)?[.!:]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:(?:certainly|sure|of course|absolutely)[,\s]+)?"
        r"(?:here (?:is|are)|here's)\s+(?:the\s+)?"
        r"(?:information|details|answer|what i found|"
        r"what you (?:requested|asked for)|results)?[.!:]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:hope this helps|let me know if you need (?:anything|more)(?: help)?|"
        r"thank you|thanks|you're welcome|you are welcome)[.!:]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?(?:refer|see|check|consult|note|visit|read|click)"
        r"\s+(?:to\s+)?(?:the\s+)?(?:documentation|docs|guide|manual|link|below|above)"
        r"\b.*[.!:]?$",
        re.IGNORECASE,
    ),
]

# Regex patterns for epistemic hedge prefixes and embedded hedges
_LEADING_HEDGE_PATTERNS = [
    re.compile(
        r"^(?:i|we)\s+(?:think|believe|assume|suppose|suspect|guess|presume|feel)"
        r"(?:\s+that)?\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:in\s+my\s+(?:opinion|view)|to\s+my\s+knowledge|as\s+far\s+as\s+i\s+know)[,\s]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^it\s+(?:is|was)\s+(?:believed|thought|claimed|said|assumed|alleged|reported|"
        r"rumored|suggested)(?:\s+that)?\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^it\s+(?:seems|appears)(?:\s+that)?\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^some\s+(?:sources|experts|people|critics|historians)?\s*"
        r"(?:claim|say|believe|suggest)(?:\s+that)?\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^according\s+to\s+[^,]+,\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:apparently|reportedly|presumably|allegedly|supposedly|possibly|perhaps)[,\s]+",
        re.IGNORECASE,
    ),
]

_EMBEDDED_HEDGE_PATTERN = re.compile(
    r",\s*(?:i\s+(?:think|believe|suppose|guess)|it\s+(?:is\s+believed|seems)|"
    r"in\s+my\s+opinion)\s*,",
    re.IGNORECASE,
)

_TRAILING_HEDGE_PATTERN = re.compile(
    r",\s*(?:i\s+(?:think|believe|suppose|guess)|it\s+(?:is\s+believed|seems)|"
    r"apparently|reportedly|allegedly)\.?$",
    re.IGNORECASE,
)

# Regex patterns for clause splitting
_CLAUSE_CONJUNCTION_EXPLICIT_SUBJECT = re.compile(
    r"^(.*?)(?:,\s*|\s+)(?:and|but|whereas|while|yet)\s+"
    r"((?:it|he|she|they|this|that|these|those)\b.*|"
    r"[A-Z][a-zA-Z0-9_\-]*\s+(?:is|was|are|were|has|have|had|will|would|can|could|did|"
    r"[a-z]+ed|[a-z]+s)\b.*)$",
    re.IGNORECASE,
)

_CLAUSE_COMPOUND_PREDICATE = re.compile(
    r"^(.+?)\s+(?:,\s*)?(?:and|but)\s+"
    r"((?:is|was|are|were|has|have|had|will|would|can|could|did|became|stands|lies|"
    r"serves|features|contains|located|[a-z]+ed)\b.*)$",
    re.IGNORECASE,
)

_VERB_SPLIT_PATTERN = re.compile(
    r"\s+(?:is|was|are|were|has|have|had|will|would|can|could|did|became|"
    r"stands|lies|serves|features|contains|began|[a-z]+ed)\b",
    re.IGNORECASE,
)

_NLTK_RESOURCES_INITIALIZED = False


def _ensure_nltk_resources() -> None:
    """Ensure required NLTK tokenizers are loaded once without per-call overhead."""
    global _NLTK_RESOURCES_INITIALIZED
    if _NLTK_RESOURCES_INITIALIZED:
        return
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:  # pragma: no cover
            pkg = resource.split("/")[1]
            nltk.download(pkg, quiet=True)
    _NLTK_RESOURCES_INITIALIZED = True


class ClaimSegmenter:
    """Extracts atomic, verifiable claim propositions from text deterministically."""

    @classmethod
    def extract_claims(cls, text: str) -> List[AtomicClaim]:
        """Extract atomic claims from text without invoking LLM passes."""
        if not text or not text.strip():
            return []

        # 1. Clean out markdown code blocks
        cleaned_text = re.sub(r"```[\s\S]*?```", "", text).strip()
        if not cleaned_text:
            return []

        # 2. Strip markdown headers and bullet/numbered list prefixes
        cleaned_text = re.sub(r"(?m)^\s*#{1,6}\s+", "", cleaned_text)
        cleaned_text = re.sub(r"(?m)^\s*(?:[-*+•]|\d+[.)])\s+", "", cleaned_text)

        # 3. Ensure NLTK resources and tokenize into sentences
        _ensure_nltk_resources()
        sentences = nltk.sent_tokenize(cleaned_text)

        claims: List[AtomicClaim] = []
        claim_counter = 0

        for raw_sent in sentences:
            sent = raw_sent.strip()
            if not sent:
                continue

            # 4. Filter non-factual interrogatives (questions)
            if sent.endswith("?"):
                continue

            # 5. Filter conversational filler / greetings
            if cls._is_conversational_filler(sent):
                continue

            # 6. Detect and process epistemic hedges (REQ-VER-03)
            processed_text, has_hedge = cls._process_epistemic_hedges(sent)

            # 7. Split compound clauses into atomic propositions (REQ-VER-02)
            clause_propositions = cls._split_into_clauses(processed_text)

            for prop in clause_propositions:
                claims.append(
                    AtomicClaim(
                        claim_id=claim_counter,
                        claim_text=prop,
                        original_sentence=sent,
                        has_epistemic_hedge=has_hedge,
                    )
                )
                claim_counter += 1

        return claims

    @classmethod
    def _split_into_clauses(cls, sentence: str) -> List[str]:
        """Split a sentence into independent clauses or return the single proposition."""
        text = sentence.strip()

        # Case 1: Semicolon separation
        if ";" in text:
            clauses = [clause.strip() for clause in text.split(";") if clause.strip()]
            if len(clauses) > 1 and all(len(c.split()) >= 3 for c in clauses):
                return [cls._format_claim(c) for c in clauses]

        # Case 2: Compound predicate sharing subject
        match_compound = _CLAUSE_COMPOUND_PREDICATE.match(text)
        if match_compound:
            part1 = match_compound.group(1).strip()
            part2 = match_compound.group(2).strip()
            subject = cls._extract_subject(part1)
            if subject and len(part1.split()) >= 3 and len(part2.split()) >= 2:
                clause2 = f"{subject} {part2}"
                return [cls._format_claim(part1), cls._format_claim(clause2)]

        # Case 3: Coordinating conjunction with explicit subject
        match_explicit = _CLAUSE_CONJUNCTION_EXPLICIT_SUBJECT.match(text)
        if match_explicit:
            part1 = match_explicit.group(1).strip()
            part2 = match_explicit.group(2).strip()
            if len(part1.split()) >= 3 and len(part2.split()) >= 3:
                return [cls._format_claim(part1), cls._format_claim(part2)]

        return [cls._format_claim(text)]

    @classmethod
    def _extract_subject(cls, clause_text: str) -> Optional[str]:
        """Extract the leading subject noun phrase preceding the first predicate verb."""
        match = _VERB_SPLIT_PATTERN.search(clause_text)
        if match:
            candidate = clause_text[: match.start()].strip()
            if candidate and len(candidate.split()) <= 6:
                return candidate
        return None

    @classmethod
    def _format_claim(cls, text: str) -> str:
        """Format an atomic claim string with proper capitalization and trailing period."""
        clean = text.strip()
        if not clean:
            return ""
        clean = clean[0].upper() + clean[1:]
        if not clean.endswith((".", "!", "?")):
            clean += "."
        return clean

    @classmethod
    def _process_epistemic_hedges(cls, sentence: str) -> tuple[str, bool]:
        """Detect and strip epistemic hedges, producing an atomic hypothesis proposition."""
        text = sentence.strip()
        has_hedge = False

        # Check leading hedges
        for pattern in _LEADING_HEDGE_PATTERNS:
            match = pattern.match(text)
            if match:
                has_hedge = True
                text = text[match.end() :].strip()
                break

        # Check embedded/parenthetical hedges
        if _EMBEDDED_HEDGE_PATTERN.search(text):
            has_hedge = True
            text = _EMBEDDED_HEDGE_PATTERN.sub(" ", text).strip()

        # Check trailing hedges
        match_trailing = _TRAILING_HEDGE_PATTERN.search(text)
        if match_trailing:
            has_hedge = True
            text = _TRAILING_HEDGE_PATTERN.sub(".", text).strip()

        if has_hedge and text:
            # Normalize whitespace and format
            text = re.sub(r"\s+", " ", text)
            text = cls._format_claim(text)

        return text, has_hedge

    @classmethod
    def _is_conversational_filler(cls, sentence: str) -> bool:
        """Check if a sentence is non-factual conversational preamble or sign-off."""
        clean = sentence.strip()
        for pattern in _CONVERSATIONAL_FILLER_PATTERNS:
            if pattern.match(clean):
                return True
        return False
