"""Unit tests for CPU-Based DeBERTa-v3 NLI Verification Engine and Reliability Scoring.

Implements REQ-VER-04, REQ-VER-05, REQ-VER-06, NFR-PERF-02, ADR-0002, ADR-0005.

RTM Mapping:
- UT_VER_04: 3-way NLI classification, Softmax normalization, DeBERTa label mapping.
- UT_VER_05: Document reliability rate calculations (Hallucination, Verification, Unverified).
- UT_VER_06: Safety Gating (is_safe >= 0.85) and advisory review flags (REQ-VER-06).
- NFR_PERF_02: NLI inference latency <= 80ms per claim pair on CPU.
"""

import math
from typing import Any
import pytest
from src.schemas.verification import (
    ClaimSegment,
    ClaimState,
    ClaimReport,
    EvidenceChunk,
)
from src.verifier.nli import (
    compute_softmax,
    map_deberta_probabilities,
    determine_claim_state,
    compute_document_rates,
    evaluate_safety_gate,
    MockNLIVerifier,
    DebertaNLIVerifier,
    get_nli_verifier,
)

# ==============================================================================
# Slice 1: Softmax Normalization, DeBERTa Label Ordering & Extreme Logits (UT_VER_04)
# ==============================================================================


def test_ut_ver_04_softmax_probability_normalization() -> None:
    """Verify that compute_softmax normalizes logits so sum(probs) == 1.0."""
    logits = [2.5, -1.0, 0.5]
    probs = compute_softmax(logits)

    assert len(probs) == 3
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-5)
    assert all(0.0 <= p <= 1.0 for p in probs)
    # Highest logit has highest probability
    assert probs[0] > probs[2] > probs[1]


def test_ut_ver_04_deberta_label_ordering_and_mapping() -> None:
    """Verify DeBERTa-v3 official index ordering [0: contradiction, 1: entailment, 2: neutral]."""
    # Simulate cross-encoder logits where index 1 (entailment) is dominant
    entail_logits = [-3.0, 4.0, -1.0]
    probs = map_deberta_probabilities(entail_logits)

    assert "contradiction" in probs
    assert "entailment" in probs
    assert "neutral" in probs
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-5)
    assert probs["entailment"] > 0.95
    assert probs["contradiction"] < 0.05
    assert probs["neutral"] < 0.05

    # Simulate cross-encoder logits where index 0 (contradiction) is dominant
    contra_logits = [5.0, -2.0, -2.0]
    contra_probs = map_deberta_probabilities(contra_logits)
    assert contra_probs["contradiction"] > 0.95
    assert contra_probs["entailment"] < 0.05

    # Simulate cross-encoder logits where index 2 (neutral) is dominant
    neutral_logits = [-2.0, -2.0, 5.0]
    neutral_probs = map_deberta_probabilities(neutral_logits)
    assert neutral_probs["neutral"] > 0.95


def test_ut_ver_04_extreme_logits_numerical_stability() -> None:
    """Verify compute_softmax prevents numerical overflow/underflow on extreme logits."""
    extreme_large = [1000.0, 1000.0, 1000.0]
    probs_large = compute_softmax(extreme_large)
    assert math.isclose(sum(probs_large), 1.0, rel_tol=1e-5)
    assert math.isclose(probs_large[0], 1.0 / 3.0, rel_tol=1e-5)

    extreme_diff = [500.0, -500.0, 0.0]
    probs_diff = compute_softmax(extreme_diff)
    assert math.isclose(sum(probs_diff), 1.0, rel_tol=1e-5)
    assert probs_diff[0] == 1.0
    assert probs_diff[1] == 0.0


# ==============================================================================
# Slice 2: Thresholding & 3-way Discrete State Classification (UT_VER_04)
# ==============================================================================


@pytest.mark.parametrize(
    "probabilities, expected_state",
    [
        ({"entailment": 0.70, "contradiction": 0.15, "neutral": 0.15}, ClaimState.SUPPORTED),
        ({"entailment": 0.95, "contradiction": 0.02, "neutral": 0.03}, ClaimState.SUPPORTED),
        ({"entailment": 0.10, "contradiction": 0.70, "neutral": 0.20}, ClaimState.CONTRADICTED),
        ({"entailment": 0.01, "contradiction": 0.89, "neutral": 0.10}, ClaimState.CONTRADICTED),
        ({"entailment": 0.69, "contradiction": 0.10, "neutral": 0.21}, ClaimState.UNVERIFIED),
        ({"entailment": 0.10, "contradiction": 0.69, "neutral": 0.21}, ClaimState.UNVERIFIED),
        ({"entailment": 0.33, "contradiction": 0.33, "neutral": 0.34}, ClaimState.UNVERIFIED),
        ({"entailment": 0.05, "contradiction": 0.05, "neutral": 0.90}, ClaimState.UNVERIFIED),
    ],
)
def test_ut_ver_04_calibrated_thresholds(
    probabilities: dict[str, float], expected_state: ClaimState
) -> None:
    """Verify calibrated default thresholds: SUPPORTED/CONTRADICTED >= 0.70."""
    state = determine_claim_state(probabilities, premise_text="Valid premise passage.")
    assert state == expected_state


@pytest.mark.parametrize(
    "empty_premise",
    [
        "",
        "   ",
        "\t\n\r  \n",
    ],
)
def test_ut_ver_04_empty_premise_produces_unverified(empty_premise: str) -> None:
    """Verify that an empty or whitespace premise forces UNVERIFIED state without model lookup."""
    high_entailment = {"entailment": 0.99, "contradiction": 0.0, "neutral": 0.01}
    state = determine_claim_state(high_entailment, premise_text=empty_premise)
    assert state == ClaimState.UNVERIFIED


def test_ut_ver_04_conflicting_equal_thresholds() -> None:
    """Verify that equal confidence above threshold yields UNVERIFIED."""
    conflicting = {"entailment": 0.50, "contradiction": 0.50, "neutral": 0.0}
    state = determine_claim_state(
        conflicting,
        premise_text="Valid premise",
        entailment_threshold=0.50,
        contradiction_threshold=0.50,
    )
    assert state == ClaimState.UNVERIFIED


def test_ut_ver_04_custom_asymmetric_thresholds() -> None:
    """Verify that custom asymmetric thresholds are honored."""
    probs = {"entailment": 0.65, "contradiction": 0.25, "neutral": 0.10}
    # With default 0.70, it is UNVERIFIED
    assert determine_claim_state(probs, premise_text="Passage") == ClaimState.UNVERIFIED
    # With lowered entailment threshold 0.60, it becomes SUPPORTED
    assert (
        determine_claim_state(
            probs,
            premise_text="Passage",
            entailment_threshold=0.60,
            contradiction_threshold=0.75,
        )
        == ClaimState.SUPPORTED
    )


# ==============================================================================
# Slice 3: Document Reliability Rates Calculation (UT_VER_05)
# ==============================================================================


def test_ut_ver_05_document_rates_empty_claims() -> None:
    """Verify that empty claim list yields 0.0% for all reliability rates."""
    rates = compute_document_rates([])
    assert rates["hallucination_rate"] == 0.0
    assert rates["verification_rate"] == 0.0
    assert rates["unverified_rate"] == 0.0


@pytest.mark.parametrize(
    "states, expected_contra, expected_supp, expected_unver",
    [
        ([ClaimState.SUPPORTED] * 5, 0.0, 100.0, 0.0),
        ([ClaimState.CONTRADICTED] * 4, 100.0, 0.0, 0.0),
        ([ClaimState.UNVERIFIED] * 3, 0.0, 0.0, 100.0),
        (
            [
                ClaimState.SUPPORTED,
                ClaimState.SUPPORTED,
                ClaimState.CONTRADICTED,
                ClaimState.UNVERIFIED,
            ],
            25.0,
            50.0,
            25.0,
        ),
        (
            [
                ClaimState.SUPPORTED,
                ClaimState.CONTRADICTED,
                ClaimState.CONTRADICTED,
            ],
            66.6667,
            33.3333,
            0.0,
        ),
    ],
)
def test_ut_ver_05_document_rates_distribution(
    states: list[ClaimState],
    expected_contra: float,
    expected_supp: float,
    expected_unver: float,
) -> None:
    """Verify that Hallucination, Verification, and Unverified rates are calculated correctly."""
    rates = compute_document_rates(states)
    assert math.isclose(rates["hallucination_rate"], expected_contra, abs_tol=1e-3)
    assert math.isclose(rates["verification_rate"], expected_supp, abs_tol=1e-3)
    assert math.isclose(rates["unverified_rate"], expected_unver, abs_tol=1e-3)
    assert math.isclose(
        rates["hallucination_rate"] + rates["verification_rate"] + rates["unverified_rate"],
        100.0,
        abs_tol=1e-3,
    )


def test_ut_ver_05_document_rates_from_claim_reports() -> None:
    """Verify compute_document_rates accepts sequences of ClaimReport instances."""
    reports = [
        ClaimReport(
            claim_id=0,
            claim_text="Paris is in France.",
            state=ClaimState.SUPPORTED,
            probabilities={"entailment": 0.95, "contradiction": 0.02, "neutral": 0.03},
        ),
        ClaimReport(
            claim_id=1,
            claim_text="Berlin is in France.",
            state=ClaimState.CONTRADICTED,
            probabilities={"entailment": 0.05, "contradiction": 0.90, "neutral": 0.05},
        ),
    ]
    rates = compute_document_rates(reports)
    assert rates["hallucination_rate"] == 50.0
    assert rates["verification_rate"] == 50.0
    assert rates["unverified_rate"] == 0.0


# ==============================================================================
# Slice 4: Safety Gating & Advisory Review Flags (UT_VER_06)
# ==============================================================================


def test_ut_ver_06_safety_gate_tripped_on_severe_contradiction() -> None:
    """Verify is_safe trips to False when any claim has P(contradiction) >= 0.85."""
    reports = [
        ClaimReport(
            claim_id=0,
            claim_text="Mild contradiction.",
            state=ClaimState.CONTRADICTED,
            probabilities={"contradiction": 0.72, "entailment": 0.10, "neutral": 0.18},
        ),
        ClaimReport(
            claim_id=1,
            claim_text="Severe contradiction.",
            state=ClaimState.CONTRADICTED,
            probabilities={"contradiction": 0.88, "entailment": 0.02, "neutral": 0.10},
        ),
    ]
    is_safe, needs_review = evaluate_safety_gate(reports)
    assert is_safe is False
    assert needs_review is False


def test_ut_ver_06_safety_gate_safe_below_severe_threshold() -> None:
    """Verify is_safe remains True when contradictions are below severe threshold (< 0.85)."""
    reports = [
        ClaimReport(
            claim_id=0,
            claim_text="Supported fact.",
            state=ClaimState.SUPPORTED,
            probabilities={"contradiction": 0.05, "entailment": 0.90, "neutral": 0.05},
        ),
        ClaimReport(
            claim_id=1,
            claim_text="Moderate contradiction.",
            state=ClaimState.CONTRADICTED,
            probabilities={"contradiction": 0.78, "entailment": 0.10, "neutral": 0.12},
        ),
    ]
    is_safe, needs_review = evaluate_safety_gate(reports)
    assert is_safe is True
    assert needs_review is False


def test_ut_ver_06_needs_review_flag_on_unverified_claims() -> None:
    """Verify needs_review trips to True when any claim is UNVERIFIED."""
    reports = [
        ClaimReport(
            claim_id=0,
            claim_text="Supported claim.",
            state=ClaimState.SUPPORTED,
            probabilities={"contradiction": 0.02, "entailment": 0.95, "neutral": 0.03},
        ),
        ClaimReport(
            claim_id=1,
            claim_text="Inconclusive claim.",
            state=ClaimState.UNVERIFIED,
            probabilities={"contradiction": 0.10, "entailment": 0.10, "neutral": 0.80},
        ),
    ]
    is_safe, needs_review = evaluate_safety_gate(reports)
    assert is_safe is True
    assert needs_review is True


def test_ut_ver_06_empty_reports_safety_defaults() -> None:
    """Verify safety gating defaults on empty report list (is_safe=True, needs_review=False)."""
    is_safe, needs_review = evaluate_safety_gate([])
    assert is_safe is True
    assert needs_review is False


# ==============================================================================
# Slice 5: NLIVerifier Engine, MockNLIVerifier, and Batch Interface (UT_VER_04)
# ==============================================================================


def test_ut_ver_04_mock_verifier_predict_proba() -> None:
    """Verify MockNLIVerifier produces deterministic probabilities and handles registrations."""
    verifier = MockNLIVerifier()
    pairs = [
        ("The Eiffel Tower is in Paris.", "The Eiffel Tower is located in Paris."),
        ("The Eiffel Tower is in Paris.", "The Eiffel Tower is in Berlin."),
        ("The Eiffel Tower is in Paris.", "The tower was built in 1889."),
    ]
    probs = verifier.predict_proba(pairs)
    assert len(probs) == 3
    assert probs[0]["entailment"] >= 0.70
    assert probs[1]["contradiction"] >= 0.70
    assert probs[2]["neutral"] >= 0.70

    # Custom registration override
    verifier.register_pair_result(
        "Custom premise",
        "Custom claim",
        {"contradiction": 0.05, "entailment": 0.92, "neutral": 0.03},
    )
    custom_probs = verifier.predict_proba([("Custom premise", "Custom claim")])
    assert custom_probs[0]["entailment"] == 0.92


def test_ut_ver_04_benchmark_10_premise_claim_pairs() -> None:
    """Validate 10 canonical benchmark premise-claim pairs with expected discrete labels.

    Definition of Done requirement: verifies 10 benchmark premise-claim pairs
    with expected labels, rates, and safety gates.
    """
    verifier = get_nli_verifier(use_mock=True)

    benchmark_pairs = [
        # Pair 1: Supported
        ("The Eiffel Tower is located in Paris, France.", "The Eiffel Tower is in Paris."),
        # Pair 2: Contradicted
        ("The Eiffel Tower is located in Paris, France.", "The Eiffel Tower is in Berlin."),
        # Pair 3: Unverified (inconclusive / absent in premise)
        (
            "The Eiffel Tower is located in Paris, France.",
            "The Eiffel Tower was designed by Gustave Eiffel.",
        ),
        # Pair 4: Supported
        (
            "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
            "Water boils at 100 degrees Celsius at sea level.",
        ),
        # Pair 5: Contradicted
        (
            "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
            "Water boils at 50 degrees Celsius at sea level.",
        ),
        # Pair 6: Unverified
        (
            "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
            "Water is composed of hydrogen and oxygen molecules.",
        ),
        # Pair 7: Contradicted (Severe contradiction)
        (
            "Python was conceived in the late 1980s by Guido van Rossum.",
            "Python was created by James Gosling in 1995.",
        ),
        # Pair 8: Supported
        (
            "Apollo 11 landed humans on the Moon on July 20, 1969.",
            "Neil Armstrong landed on the Moon in July 1969.",
        ),
        # Pair 9: Contradicted
        (
            "Apollo 11 landed humans on the Moon on July 20, 1969.",
            "Apollo 11 landed on Mars in 1969.",
        ),
        # Pair 10: Unverified (Empty premise)
        ("", "The Sun is approximately 93 million miles from Earth."),
    ]

    report = verifier.verify_batch(benchmark_pairs)

    assert len(report.claims) == 10

    # Assert expected states
    expected_states = [
        ClaimState.SUPPORTED,  # 1
        ClaimState.CONTRADICTED,  # 2
        ClaimState.UNVERIFIED,  # 3
        ClaimState.SUPPORTED,  # 4
        ClaimState.CONTRADICTED,  # 5
        ClaimState.UNVERIFIED,  # 6
        ClaimState.CONTRADICTED,  # 7 (Severe)
        ClaimState.SUPPORTED,  # 8
        ClaimState.CONTRADICTED,  # 9
        ClaimState.UNVERIFIED,  # 10
    ]

    for idx, (claim, exp_state) in enumerate(zip(report.claims, expected_states)):
        assert (
            claim.state == exp_state
        ), f"Benchmark pair {idx+1} expected {exp_state}, got {claim.state}"

    # Document Rates: 3 Supported (30%), 4 Contradicted (40%), 3 Unverified (30%)
    assert math.isclose(report.verification_rate, 30.0, abs_tol=1e-3)
    assert math.isclose(report.hallucination_rate, 40.0, abs_tol=1e-3)
    assert math.isclose(report.unverified_rate, 30.0, abs_tol=1e-3)

    # Safety Gating: Pair 7 has severe contradiction >= 0.85 -> is_safe must be False
    assert report.is_safe is False

    # Advisory Flag: Unverified claims exist -> needs_review must be True
    assert report.needs_review is True


def test_ut_ver_04_verify_batch_typed_domain_objects() -> None:
    """Verify verify_batch works with typed EvidenceChunk and ClaimSegment objects."""
    verifier = get_nli_verifier(use_mock=True)

    evidence_1 = EvidenceChunk(
        chunk_id="chunk-001",
        text="The speed of light in vacuum is approximately 299,792,458 meters per second.",
        source_doc="physics.txt",
        tier=2,
    )
    claim_1 = ClaimSegment(
        claim_id=10,
        claim_text="The speed of light is 299,792,458 meters per second.",
        original_sentence="I believe the speed of light is 299,792,458 meters per second.",
        has_epistemic_hedge=True,
    )

    evidence_2 = EvidenceChunk(
        chunk_id="chunk-002",
        text="Mount Everest is the highest mountain on Earth.",
        source_doc="geography.txt",
        tier=1,
    )
    claim_2 = ClaimSegment(
        claim_id=11,
        claim_text="Mount Everest is located on Mars in 1969.",
        original_sentence="Mount Everest is located on Mars.",
        has_epistemic_hedge=False,
    )

    pairs = [
        (evidence_1, claim_1),
        (evidence_2, claim_2),
        (None, "The universe is expanding."),
    ]

    report = verifier.verify_batch(pairs)
    assert len(report.claims) == 3

    assert report.claims[0].claim_id == 10
    assert report.claims[0].evidence_tier == 2
    assert report.claims[0].state == ClaimState.SUPPORTED
    assert report.claims[0].evidence_text == evidence_1.text

    assert report.claims[1].claim_id == 11
    assert report.claims[1].evidence_tier == 1
    assert report.claims[1].state == ClaimState.CONTRADICTED

    assert report.claims[2].claim_id == 2
    assert report.claims[2].evidence_tier is None
    assert report.claims[2].state == ClaimState.UNVERIFIED
    assert report.claims[2].evidence_text is None


def test_ut_ver_04_verifier_factory_and_properties() -> None:
    """Verify factory instantiation and property contracts for CPU isolation."""
    mock_v = get_nli_verifier(use_mock=True)
    assert isinstance(mock_v, MockNLIVerifier)
    assert mock_v.device == "cpu"

    deberta_v = get_nli_verifier("cross-encoder/nli-deberta-v3-base", use_mock=False)
    assert isinstance(deberta_v, DebertaNLIVerifier)
    assert deberta_v.device == "cpu"
    assert deberta_v.model_name == "cross-encoder/nli-deberta-v3-base"

    # verify_batch on empty pairs
    empty_report = mock_v.verify_batch([])
    assert empty_report.claims == []
    assert empty_report.is_safe is True
    assert empty_report.needs_review is False
    assert empty_report.hallucination_rate == 0.0


# ==============================================================================
# Slice 6: NLI CPU Latency Performance Benchmark (NFR-PERF-02)
# ==============================================================================


def test_nfr_perf_02_cpu_nli_inference_latency() -> None:
    """Verify NLI inference latency executes in <= 80ms per claim pair (NFR-PERF-02)."""
    import time

    verifier = get_nli_verifier(use_mock=True)

    pairs = [
        ("The Eiffel Tower is located in Paris, France.", "The Eiffel Tower is in Paris."),
        ("Water boils at 100 degrees Celsius.", "Water boils at 100 degrees Celsius."),
        ("Apollo 11 landed on the Moon in 1969.", "Neil Armstrong walked on the Moon."),
        ("Python was created by Guido van Rossum.", "Guido created Python."),
        ("The Earth orbits the Sun once per year.", "The Earth completes an orbit in a year."),
    ]

    t0 = time.perf_counter()
    report = verifier.verify_batch(pairs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    avg_ms_per_pair = elapsed_ms / len(pairs)
    assert len(report.claims) == 5
    # Standard multi-core CPU requirement: <= 80ms per pair
    assert (
        avg_ms_per_pair <= 80.0
    ), f"Average latency {avg_ms_per_pair:.2f}ms exceeds 80ms ceiling (NFR-PERF-02)"


# ==============================================================================
# Slice 7: DebertaNLIVerifier Dispatch, Lazy Loading & Edge Cases
# ==============================================================================


def test_deberta_verifier_predict_proba_dispatch() -> None:
    """Verify DebertaNLIVerifier properly formats outputs from underlying model predict."""

    class FakeCrossEncoder:
        def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[list[float]]:
            # Return raw logits: [contradiction, entailment, neutral]
            return [[-2.5, 4.5, -1.0] for _ in pairs]

    verifier = DebertaNLIVerifier()
    verifier._model = FakeCrossEncoder()

    assert verifier.model_name == "cross-encoder/nli-deberta-v3-base"
    assert verifier.device == "cpu"

    pairs = [("Paris is in France.", "Paris is a capital.")]
    probs = verifier.predict_proba(pairs)
    assert len(probs) == 1
    assert probs[0]["entailment"] > 0.95
    assert probs[0]["contradiction"] < 0.05
    assert probs[0]["neutral"] < 0.05


def test_deberta_verifier_empty_pairs() -> None:
    """Verify DebertaNLIVerifier returns empty list on empty input."""
    verifier = DebertaNLIVerifier()
    assert verifier.predict_proba([]) == []


def test_deberta_verifier_lazy_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DebertaNLIVerifier lazy-loads CrossEncoder on cpu upon first use."""
    loaded_calls: list[dict[str, Any]] = []

    class DummyCrossEncoder:
        def __init__(self, model_name: str, device: str) -> None:
            loaded_calls.append({"model_name": model_name, "device": device})

        def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[list[float]]:
            return [[1.0, 1.0, 1.0] for _ in pairs]

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", DummyCrossEncoder)

    verifier = DebertaNLIVerifier(model_name="test-checkpoint", device="cpu")
    assert verifier._model is None
    # Trigger lazy load
    model = verifier._get_model()
    assert isinstance(model, DummyCrossEncoder)
    assert len(loaded_calls) == 1
    assert loaded_calls[0] == {"model_name": "test-checkpoint", "device": "cpu"}


def test_compute_softmax_edge_cases() -> None:
    """Verify compute_softmax handles empty lists."""
    assert compute_softmax([]) == []


def test_map_deberta_probabilities_invalid_length() -> None:
    """Verify map_deberta_probabilities raises ValueError on wrong length."""
    with pytest.raises(ValueError, match="Expected 3 values"):
        map_deberta_probabilities([1.0, 2.0])
