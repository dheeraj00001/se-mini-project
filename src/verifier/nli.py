"""CPU-Based DeBERTa-v3 NLI Verification Engine and Reliability Scoring.

Implements REQ-VER-04, REQ-VER-05, REQ-VER-06, NFR-PERF-02, ADR-0002, ADR-0005.

Core Responsibilities:
1. Cross-encoder NLI inference strictly on host CPU threads.
2. Softmax normalization over logits: P(entailment) + P(neutral) + P(contradiction) == 1.0.
3. Label index mapping matching HuggingFace DeBERTa: [0: contradiction, 1: entailment, 2: neutral].
4. 3-way discrete classification with calibrated thresholds (tau_e >= 0.70, tau_c >= 0.70).
5. Document reliability rate metrics (Hallucination Rate, Verification Rate, Unverified Rate).
6. Dual-flag safety gating (is_safe = False on contradiction >= 0.85; needs_review on unverified).
"""

from abc import ABC, abstractmethod
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.schemas.verification import (
    ClaimReport,
    ClaimSegment,
    ClaimState,
    DEBERTA_NLI_LABEL_ORDER,
    EvidenceChunk,
    VerificationReport,
    determine_claim_state_from_nli,
    map_nli_probabilities,
)


def compute_softmax(logits: Sequence[float]) -> List[float]:
    """Compute numerically stable Softmax probabilities from raw logits.

    Ensures sum(probabilities) == 1.0 and prevents overflow via max-subtraction.
    """
    if not logits:
        return []
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [e / total for e in exps]


def map_deberta_probabilities(
    logits_or_probs: Sequence[float],
    label_order: Sequence[str] = DEBERTA_NLI_LABEL_ORDER,
    is_logits: bool = True,
) -> Dict[str, float]:
    """Map DeBERTa output scores to explicit label names.

    HuggingFace DeBERTa-v3 cross-encoder ordering:
    - Index 0: contradiction
    - Index 1: entailment
    - Index 2: neutral

    Args:
        logits_or_probs: 3-element sequence of raw logits or normalized probabilities.
        label_order: Sequence of label strings matching output indices.
        is_logits: If True, applies Softmax prior to mapping.
    """
    if len(logits_or_probs) != len(label_order):
        raise ValueError(
            f"Expected {len(label_order)} values matching {label_order}, got {len(logits_or_probs)}"
        )
    probs = compute_softmax(logits_or_probs) if is_logits else list(logits_or_probs)
    return map_nli_probabilities(probs, label_order=label_order)


def determine_claim_state(
    probabilities: Dict[str, float],
    premise_text: Optional[str] = None,
    entailment_threshold: float = 0.70,
    contradiction_threshold: float = 0.70,
) -> ClaimState:
    """Classify an atomic claim into discrete ClaimState (REQ-VER-04, ADR-0005).

    Args:
        probabilities: Label probabilities for entailment, contradiction, neutral.
        premise_text: Optional premise passage. If empty/whitespace, returns UNVERIFIED.
        entailment_threshold: Minimum confidence for SUPPORTED (default 0.70).
        contradiction_threshold: Minimum confidence for CONTRADICTED (default 0.70).
    """
    if premise_text is not None and not premise_text.strip():
        return ClaimState.UNVERIFIED

    return determine_claim_state_from_nli(
        probabilities=probabilities,
        entailment_threshold=entailment_threshold,
        contradiction_threshold=contradiction_threshold,
    )


def compute_document_rates(
    claims: Sequence[Union[ClaimReport, ClaimState]],
) -> Dict[str, float]:
    """Calculate document-level reliability rates (REQ-VER-05).

    Formulas:
    - Hallucination Rate = (N_contradicted / N_total) * 100%
    - Verification Rate  = (N_supported / N_total) * 100%
    - Unverified Rate    = (N_unverified / N_total) * 100%

    Edge Cases:
    - N_total == 0 -> all rates 0.0%
    """
    total = len(claims)
    if total == 0:
        return {
            "hallucination_rate": 0.0,
            "verification_rate": 0.0,
            "unverified_rate": 0.0,
        }

    states = [c.state if isinstance(c, ClaimReport) else c for c in claims]
    n_contra = sum(1 for s in states if s == ClaimState.CONTRADICTED)
    n_supp = sum(1 for s in states if s == ClaimState.SUPPORTED)
    n_unver = sum(1 for s in states if s == ClaimState.UNVERIFIED)

    return {
        "hallucination_rate": round((n_contra / total) * 100.0, 4),
        "verification_rate": round((n_supp / total) * 100.0, 4),
        "unverified_rate": round((n_unver / total) * 100.0, 4),
    }


def evaluate_safety_gate(
    claims: Sequence[ClaimReport],
    severe_contradiction_threshold: float = 0.85,
) -> Tuple[bool, bool]:
    """Evaluate system safety circuit breaker and advisory review flags (REQ-VER-06, ADR-0005).

    Returns:
        (is_safe, needs_review)
        - is_safe: False if any claim has P(contradiction) >= 0.85; True otherwise.
        - needs_review: True if any claim is UNVERIFIED; False otherwise.
    """
    if not claims:
        return True, False

    is_safe = True
    needs_review = False

    for claim in claims:
        contra_p = claim.probabilities.get("contradiction", 0.0)
        if contra_p >= severe_contradiction_threshold:
            is_safe = False
        if claim.state == ClaimState.UNVERIFIED:
            needs_review = True

    return is_safe, needs_review


class BaseNLIVerifier(ABC):
    """Abstract base class for Natural Language Inference verifiers."""

    @abstractmethod
    def predict_proba(self, pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, float]]:
        """Predict 3-way NLI probability distributions for (Premise, Claim) pairs."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model identifier."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Return target device ('cpu')."""
        pass

    def verify_batch(
        self,
        pairs: Sequence[Tuple[Any, Any]],
        entailment_threshold: float = 0.70,
        contradiction_threshold: float = 0.70,
        severe_contradiction_threshold: float = 0.85,
    ) -> VerificationReport:
        """Verify a batch of premise-claim pairs, computing document rates and safety flags."""
        if not pairs:
            rates = compute_document_rates([])
            return VerificationReport(
                is_safe=True,
                needs_review=False,
                hallucination_rate=rates["hallucination_rate"],
                verification_rate=rates["verification_rate"],
                unverified_rate=rates["unverified_rate"],
                claims=[],
            )

        extracted_pairs: List[Tuple[str, str]] = []
        claim_metadata: List[Dict[str, Any]] = []

        for idx, pair in enumerate(pairs):
            premise_obj, claim_obj = pair

            # Normalize premise
            premise_text = ""
            evidence_tier: Optional[int] = None
            if isinstance(premise_obj, EvidenceChunk):
                premise_text = premise_obj.text or ""
                evidence_tier = premise_obj.tier
            elif isinstance(premise_obj, str):
                premise_text = premise_obj

            # Normalize claim
            claim_id = idx
            claim_text = ""
            if isinstance(claim_obj, ClaimSegment):
                claim_text = claim_obj.claim_text
                claim_id = claim_obj.claim_id
            elif isinstance(claim_obj, str):
                claim_text = claim_obj

            extracted_pairs.append((premise_text, claim_text))
            claim_metadata.append(
                {
                    "claim_id": claim_id,
                    "claim_text": claim_text,
                    "premise_text": premise_text,
                    "evidence_tier": evidence_tier,
                }
            )

        # Filter out empty premises from model inference to preserve CPU cycles
        pairs_to_evaluate: List[Tuple[int, Tuple[str, str]]] = []
        for i, (p, c) in enumerate(extracted_pairs):
            if p.strip() and c.strip():
                pairs_to_evaluate.append((i, (p, c)))

        probabilities_list: List[Dict[str, float]] = [
            {"contradiction": 0.0, "entailment": 0.0, "neutral": 1.0} for _ in extracted_pairs
        ]

        if pairs_to_evaluate:
            indices = [item[0] for item in pairs_to_evaluate]
            eval_pairs = [item[1] for item in pairs_to_evaluate]
            model_probs = self.predict_proba(eval_pairs)
            for idx, prob_dict in zip(indices, model_probs):
                probabilities_list[idx] = prob_dict

        # Construct ClaimReports
        reports: List[ClaimReport] = []
        for meta, probs in zip(claim_metadata, probabilities_list):
            p_text = meta["premise_text"]
            # If premise is empty, state is strictly UNVERIFIED
            if not p_text.strip():
                state = ClaimState.UNVERIFIED
            else:
                state = determine_claim_state(
                    probabilities=probs,
                    premise_text=p_text,
                    entailment_threshold=entailment_threshold,
                    contradiction_threshold=contradiction_threshold,
                )

            reports.append(
                ClaimReport(
                    claim_id=meta["claim_id"],
                    claim_text=meta["claim_text"],
                    state=state,
                    probabilities=probs,
                    evidence_tier=meta["evidence_tier"],
                    evidence_text=p_text if p_text.strip() else None,
                )
            )

        rates = compute_document_rates(reports)
        is_safe, needs_review = evaluate_safety_gate(
            reports,
            severe_contradiction_threshold=severe_contradiction_threshold,
        )

        return VerificationReport(
            is_safe=is_safe,
            needs_review=needs_review,
            hallucination_rate=rates["hallucination_rate"],
            verification_rate=rates["verification_rate"],
            unverified_rate=rates["unverified_rate"],
            claims=reports,
        )


class MockNLIVerifier(BaseNLIVerifier):
    """Deterministic, zero-latency pseudo-verifier for offline testing and benchmarking."""

    def __init__(
        self,
        model_name: str = "mock-deberta-v3-base",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._custom_responses: Dict[Tuple[str, str], Dict[str, float]] = {}

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    def register_pair_result(
        self, premise: str, claim: str, probabilities: Dict[str, float]
    ) -> None:
        """Register an explicit mock verdict probability dict for a (premise, claim) pair."""
        self._custom_responses[(premise.strip(), claim.strip())] = probabilities

    def predict_proba(self, pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, float]]:
        """Predict deterministic probabilities based on registered pairs or semantic heuristics."""
        results: List[Dict[str, float]] = []

        for premise, claim in pairs:
            p_norm = premise.strip()
            c_norm = claim.strip()

            # Check explicit mock registrations first
            if (p_norm, c_norm) in self._custom_responses:
                results.append(self._custom_responses[(p_norm, c_norm)])
                continue

            # Check for empty premise -> strictly neutral
            if not p_norm:
                results.append({"contradiction": 0.0, "entailment": 0.0, "neutral": 1.0})
                continue

            p_lower = p_norm.lower()
            c_lower = c_norm.lower()

            # Benchmark Rule 1: Contradictions (direct negation or conflicting factual predicates)
            is_contra = (
                ("berlin" in c_lower and "paris" in p_lower)
                or ("50 degrees" in c_lower and "100 degrees" in p_lower)
                or ("mars" in c_lower and ("moon" in p_lower or "earth" in p_lower))
                or ("james gosling" in c_lower and "guido van rossum" in p_lower)
                or ("not " in c_lower and "not " not in p_lower)
                or ("false" in c_lower and "true" in p_lower)
            )

            # Benchmark Rule 2: Entailment (direct factual support or token overlap)
            c_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", c_lower)) - {
                "the",
                "a",
                "an",
                "is",
                "was",
                "are",
                "were",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "and",
                "by",
            }
            p_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", p_lower))
            token_overlap = (
                len(c_tokens.intersection(p_tokens)) / len(c_tokens) if c_tokens else 0.0
            )

            is_entail = (
                ("paris" in c_lower and "paris" in p_lower and "berlin" not in c_lower)
                or (
                    "100 degrees" in c_lower
                    and "100 degrees" in p_lower
                    and "50 degrees" not in c_lower
                )
                or ("neil armstrong" in c_lower and "moon" in p_lower and "apollo 11" in p_lower)
                or (
                    "guido van rossum" in c_lower
                    and "guido van rossum" in p_lower
                    and "gosling" not in c_lower
                )
                or (c_lower in p_lower)
                or (token_overlap >= 0.70 and not is_contra)
            )

            if is_contra:
                results.append({"contradiction": 0.94, "entailment": 0.03, "neutral": 0.03})
            elif is_entail:
                results.append({"contradiction": 0.02, "entailment": 0.95, "neutral": 0.03})
            else:
                # Default inconclusive / neutral
                results.append({"contradiction": 0.08, "entailment": 0.08, "neutral": 0.84})

        return results


class DebertaNLIVerifier(BaseNLIVerifier):
    """CPU-based DeBERTa-v3 Natural Language Inference verifier.

    Executes `cross-encoder/nli-deberta-v3-base` on host CPU threads (ADR-0002).
    Preserves GPU VRAM for the generative LLM.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Optional[Any] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    def _get_model(self) -> Any:
        """Lazy load CrossEncoder onto CPU threads."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            # Strictly enforce host CPU device (ADR-0002)
            self._model = CrossEncoder(
                self._model_name,
                device="cpu",
            )
        return self._model

    def predict_proba(self, pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, float]]:
        """Run batch inference on host CPU threads and return Softmax probabilities."""
        if not pairs:
            return []

        model = self._get_model()
        # predict returns logits or probabilities based on apply_softmax
        # SentenceTransformers CrossEncoder apply_softmax=True computes softmax across labels
        raw_outputs = model.predict(
            list(pairs),
            apply_softmax=False,
            convert_to_numpy=True,
        )

        results: List[Dict[str, float]] = []
        for row in raw_outputs:
            logits = row.tolist() if hasattr(row, "tolist") else list(row)
            probs = map_deberta_probabilities(logits, is_logits=True)
            results.append(probs)

        return results


# Default facade class
NLIVerifier = DebertaNLIVerifier


def get_nli_verifier(
    model_name: str = "cross-encoder/nli-deberta-v3-base",
    device: str = "cpu",
    use_mock: bool = False,
) -> BaseNLIVerifier:
    """Factory creating an NLI verifier instance (ADR-0002).

    Args:
        model_name: HuggingFace model identifier.
        device: Target hardware device (strictly 'cpu' per ADR-0002).
        use_mock: If True, returns MockNLIVerifier for fast headless testing.
    """
    if use_mock or "mock" in model_name.lower():
        return MockNLIVerifier(model_name=model_name, device=device)
    return DebertaNLIVerifier(model_name=model_name, device=device)
