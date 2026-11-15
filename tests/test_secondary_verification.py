"""Unit and integration tests for Secondary Verification Loop & Mitigation Fallback Cascade.

Implements REQ-MIT-03, REQ-MIT-04, NFR-SAFE-01, NFR-SAFE-02, ADR-0006, BR-01, BR-02.

RTM Mapping:
- UT_MIT_03: Single secondary verification loop with recursion prevention (NFR-SAFE-02).
- UT_MIT_04: Fallback cascade to redact / block (REQ-MIT-04, NFR-SAFE-01, BR-01).
- ST_MIT_02: End-to-end secondary verification pipeline and recovery.
- ST_MIT_03: Forced fallback cascade under residual hallucinations.
"""

import pytest
from typing import Any
from fastapi import status

from src.schemas.verification import (
    ClaimReport,
    ClaimState,
    MitigationMode,
    VerificationRequest,
    VerificationResponse,
)
from src.mitigation.fallback import (
    DEFAULT_REDACTION_TOKEN,
    DEFAULT_REFUSAL_MESSAGE,
    FallbackEngine,
    MitigationBlockedError,
    redact_contradicted_claims,
)
from src.mitigation.generator import MockGenerator
from src.mitigation.pipeline import MitigationPipeline
from src.mitigation.rewriter import TargetedRewriter
from src.verifier.nli import MockNLIVerifier
from src.verifier.orchestrator import WaterfallOrchestrator

# ==============================================================================
# Slice 1: Fallback Cascade Engine (UT_MIT_04 / REQ-MIT-04 / BR-01)
# ==============================================================================


def test_ut_mit_04_redact_contradicted_claims_basic() -> None:
    """Verify surgical replacement of contradicted sentences with standard token."""
    text = (
        "Paris is the capital of France. "
        "The moon is made of green cheese. "
        "Water boils at 100 degrees Celsius."
    )
    claims = [
        ClaimReport(
            claim_id=0,
            claim_text="Paris is the capital of France",
            original_sentence="Paris is the capital of France.",
            state=ClaimState.SUPPORTED,
            probabilities={"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02},
            evidence_tier=1,
            evidence_text="Paris is the capital of France.",
        ),
        ClaimReport(
            claim_id=1,
            claim_text="The moon is made of green cheese",
            original_sentence="The moon is made of green cheese.",
            state=ClaimState.CONTRADICTED,
            probabilities={"entailment": 0.01, "neutral": 0.04, "contradiction": 0.95},
            evidence_tier=2,
            evidence_text="The moon's surface is composed of silicate rock and dust.",
        ),
        ClaimReport(
            claim_id=2,
            claim_text="Water boils at 100 degrees Celsius",
            original_sentence="Water boils at 100 degrees Celsius.",
            state=ClaimState.SUPPORTED,
            probabilities={"entailment": 0.92, "neutral": 0.05, "contradiction": 0.03},
            evidence_tier=1,
            evidence_text="At standard pressure, water boils at 100 degrees Celsius.",
        ),
    ]

    redacted = redact_contradicted_claims(text, claims)

    assert DEFAULT_REDACTION_TOKEN in redacted
    assert "Paris is the capital of France." in redacted
    assert "Water boils at 100 degrees Celsius." in redacted
    assert "The moon is made of green cheese." not in redacted


def test_ut_mit_04_redact_preserves_unverified_claims_br01() -> None:
    """Enforce BR-01: UNVERIFIED claims must NEVER be redacted or masked."""
    text = (
        "Albert Einstein won the Nobel Prize in Physics. "
        "Aliens visited Earth in 1947. "
        "The earth is completely flat."
    )
    claims = [
        ClaimReport(
            claim_id=0,
            claim_text="Albert Einstein won the Nobel Prize in Physics",
            original_sentence="Albert Einstein won the Nobel Prize in Physics.",
            state=ClaimState.SUPPORTED,
            probabilities={"entailment": 0.98, "neutral": 0.01, "contradiction": 0.01},
        ),
        ClaimReport(
            claim_id=1,
            claim_text="Aliens visited Earth in 1947",
            original_sentence="Aliens visited Earth in 1947.",
            state=ClaimState.UNVERIFIED,
            probabilities={"entailment": 0.10, "neutral": 0.80, "contradiction": 0.10},
        ),
        ClaimReport(
            claim_id=2,
            claim_text="The earth is completely flat",
            original_sentence="The earth is completely flat.",
            state=ClaimState.CONTRADICTED,
            probabilities={"entailment": 0.01, "neutral": 0.02, "contradiction": 0.97},
            evidence_tier=1,
            evidence_text="Earth is an oblate spheroid.",
        ),
    ]

    redacted = redact_contradicted_claims(text, claims)

    # Contradicted claim must be redacted
    assert DEFAULT_REDACTION_TOKEN in redacted
    assert "The earth is completely flat." not in redacted

    # BR-01: Supported and Unverified must remain strictly preserved!
    assert "Albert Einstein won the Nobel Prize in Physics." in redacted
    assert "Aliens visited Earth in 1947." in redacted


def test_ut_mit_04_redact_no_contradictions_and_empty_text() -> None:
    """Verify redact returns unchanged text when no contradictions exist or text is empty."""
    text = "Everything in this statement is supported."
    claims = [
        ClaimReport(
            claim_id=0,
            claim_text=text,
            state=ClaimState.SUPPORTED,
            probabilities={"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02},
        )
    ]
    assert redact_contradicted_claims(text, claims) == text
    assert redact_contradicted_claims("", claims) == ""
    assert redact_contradicted_claims("   ", claims) == "   "


def test_ut_mit_04_block_raises_mitigation_blocked_error() -> None:
    """Verify FallbackEngine raises MitigationBlockedError (HTTP 422) on block."""
    engine = FallbackEngine()
    dummy_response = VerificationResponse(
        is_safe=False,
        needs_review=False,
        hallucination_rate=100.0,
        verification_rate=0.0,
        unverified_rate=0.0,
        claims=[],
        sanitized_text="Harmful or contradicted statement",
    )

    with pytest.raises(MitigationBlockedError) as exc_info:
        engine.block(response=dummy_response, raise_exception=True)

    err = exc_info.value
    assert err.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert err.detail == DEFAULT_REFUSAL_MESSAGE
    assert err.response is dummy_response


def test_ut_mit_04_block_returns_canonical_refusal_string() -> None:
    """Verify FallbackEngine returns canonical refusal text when raise_exception=False."""
    engine = FallbackEngine()
    result = engine.block(raise_exception=False)
    assert result == DEFAULT_REFUSAL_MESSAGE


def test_ut_mit_04_fallback_engine_execute_dispatch() -> None:
    """Verify FallbackEngine dispatch for REDACT and BLOCK modes."""
    engine = FallbackEngine()
    text = "The earth is flat."
    claims = [
        ClaimReport(
            claim_id=0,
            claim_text="The earth is flat",
            original_sentence="The earth is flat.",
            state=ClaimState.CONTRADICTED,
            probabilities={"entailment": 0.0, "neutral": 0.05, "contradiction": 0.95},
        )
    ]

    # Test REDACT dispatch
    redacted = engine.execute_fallback(
        mode=MitigationMode.REDACT,
        text=text,
        claims=claims,
    )
    assert DEFAULT_REDACTION_TOKEN in redacted

    # Test BLOCK dispatch without exception
    blocked = engine.execute_fallback(
        mode=MitigationMode.BLOCK,
        text=text,
        claims=claims,
        raise_on_block=False,
    )
    assert blocked == DEFAULT_REFUSAL_MESSAGE

    # Test BLOCK dispatch with exception
    with pytest.raises(MitigationBlockedError):
        engine.execute_fallback(
            mode=MitigationMode.BLOCK,
            text=text,
            claims=claims,
            raise_on_block=True,
        )


# ==============================================================================
# Slice 2: Secondary Verification Pipeline (UT_MIT_03 / NFR-SAFE-02 / REQ-MIT-03)
# ==============================================================================


class FakeOrchestrator:
    """Configurable fake orchestrator for testing pipeline verification loops."""

    def __init__(self, responses: list[VerificationResponse]) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.calls: list[VerificationRequest] = []

    def verify(
        self, request: VerificationRequest, _is_secondary: bool = False
    ) -> VerificationResponse:
        self.call_count += 1
        self.calls.append(request)
        if self.responses:
            return self.responses.pop(0)
        return VerificationResponse(
            is_safe=True,
            needs_review=False,
            hallucination_rate=0.0,
            verification_rate=100.0,
            unverified_rate=0.0,
            claims=[],
            sanitized_text=request.text,
        )


def test_ut_mit_03_secondary_verification_pass_clean_approval() -> None:
    """Verify that clean secondary verification approves and returns rewritten response."""
    original_text = "The capital of France is Lyon."
    rewritten_text = "The capital of France is Paris."

    initial_claims = [
        ClaimReport(
            claim_id=0,
            claim_text="The capital of France is Lyon",
            original_sentence=original_text,
            state=ClaimState.CONTRADICTED,
            probabilities={"entailment": 0.05, "neutral": 0.05, "contradiction": 0.90},
            evidence_tier=1,
            evidence_text="The capital of France is Paris.",
        )
    ]
    initial_response = VerificationResponse(
        is_safe=False,
        needs_review=False,
        hallucination_rate=100.0,
        verification_rate=0.0,
        unverified_rate=0.0,
        claims=initial_claims,
        sanitized_text=original_text,
    )

    # Mock generator configured to rewrite Lyon -> Paris
    mock_gen = MockGenerator()
    mock_gen.register_response("CONTRADICTED CLAIM", rewritten_text)
    rewriter = TargetedRewriter(generator=mock_gen)

    # Secondary verifier returns 100% supported on rewritten text
    secondary_claim = ClaimReport(
        claim_id=0,
        claim_text="The capital of France is Paris",
        original_sentence=rewritten_text,
        state=ClaimState.SUPPORTED,
        probabilities={"entailment": 0.98, "neutral": 0.01, "contradiction": 0.01},
        evidence_tier=1,
        evidence_text="The capital of France is Paris.",
    )
    secondary_response = VerificationResponse(
        is_safe=True,
        needs_review=False,
        hallucination_rate=0.0,
        verification_rate=100.0,
        unverified_rate=0.0,
        claims=[secondary_claim],
        sanitized_text=rewritten_text,
    )

    fake_orch = FakeOrchestrator([secondary_response])
    pipeline = MitigationPipeline(
        rewriter=rewriter,
        orchestrator=fake_orch,
    )

    req = VerificationRequest(
        text=original_text,
        mitigation_mode=MitigationMode.REWRITE,
    )
    res = pipeline.execute(req, initial_response)

    # Assert approved rewrite
    assert res.is_safe is True
    assert res.sanitized_text == rewritten_text
    assert res.verification_rate == 100.0
    assert fake_orch.call_count == 1
    assert fake_orch.calls[0].text == rewritten_text
    # Ensure secondary request runs with FLAG mitigation mode to prevent recursion
    assert fake_orch.calls[0].mitigation_mode == MitigationMode.FLAG


def test_ut_mit_03_secondary_verification_residual_contradiction_cascades_to_redact() -> None:
    """Verify residual contradiction after rewrite cascades to redact (REQ-MIT-04, NFR-SAFE-02)."""
    original_text = "The capital of Australia is Sydney."
    rewritten_text = "The capital of Australia is Melbourne."  # Still a hallucination!

    initial_claims = [
        ClaimReport(
            claim_id=0,
            claim_text="The capital of Australia is Sydney",
            original_sentence=original_text,
            state=ClaimState.CONTRADICTED,
            probabilities={"entailment": 0.05, "neutral": 0.05, "contradiction": 0.90},
            evidence_tier=2,
            evidence_text="Canberra is the federal capital city of Australia.",
        )
    ]
    initial_response = VerificationResponse(
        is_safe=False,
        needs_review=False,
        hallucination_rate=100.0,
        verification_rate=0.0,
        unverified_rate=0.0,
        claims=initial_claims,
        sanitized_text=original_text,
    )

    mock_gen = MockGenerator()
    mock_gen.register_response("CONTRADICTED CLAIM", rewritten_text)
    rewriter = TargetedRewriter(generator=mock_gen)

    # Secondary verification detects residual contradiction in Melbourne!
    secondary_claim = ClaimReport(
        claim_id=0,
        claim_text="The capital of Australia is Melbourne",
        original_sentence=rewritten_text,
        state=ClaimState.CONTRADICTED,
        probabilities={"entailment": 0.02, "neutral": 0.08, "contradiction": 0.90},
        evidence_tier=2,
        evidence_text="Canberra is the federal capital city of Australia.",
    )
    secondary_response = VerificationResponse(
        is_safe=False,
        needs_review=False,
        hallucination_rate=100.0,
        verification_rate=0.0,
        unverified_rate=0.0,
        claims=[secondary_claim],
        sanitized_text=rewritten_text,
    )

    fake_orch = FakeOrchestrator([secondary_response])
    pipeline = MitigationPipeline(
        rewriter=rewriter,
        orchestrator=fake_orch,
        fallback_mode=MitigationMode.REDACT,
    )

    req = VerificationRequest(
        text=original_text,
        mitigation_mode=MitigationMode.REWRITE,
    )
    res = pipeline.execute(req, initial_response)

    # Assert cascaded to redact
    assert DEFAULT_REDACTION_TOKEN in res.sanitized_text
    assert "Melbourne" not in res.sanitized_text
    # Exactly one secondary verification pass (NFR-SAFE-02)
    assert fake_orch.call_count == 1


def test_ut_mit_03_secondary_verification_residual_contradiction_cascades_to_block() -> None:
    """Verify residual contradiction with fallback_mode=BLOCK raises MitigationBlockedError."""
    original_text = "The earth is 6000 years old."
    rewritten_text = "The earth is 10000 years old."

    initial_claims = [
        ClaimReport(
            claim_id=0,
            claim_text="The earth is 6000 years old",
            original_sentence=original_text,
            state=ClaimState.CONTRADICTED,
            probabilities={"contradiction": 0.99, "entailment": 0.0, "neutral": 0.01},
        )
    ]
    initial_response = VerificationResponse(
        is_safe=False,
        needs_review=False,
        hallucination_rate=100.0,
        verification_rate=0.0,
        unverified_rate=0.0,
        claims=initial_claims,
        sanitized_text=original_text,
    )

    mock_gen = MockGenerator()
    mock_gen.register_response("CONTRADICTED CLAIM", rewritten_text)
    rewriter = TargetedRewriter(generator=mock_gen)

    secondary_claim = ClaimReport(
        claim_id=0,
        claim_text="The earth is 10000 years old",
        original_sentence=rewritten_text,
        state=ClaimState.CONTRADICTED,
        probabilities={"contradiction": 0.95, "entailment": 0.01, "neutral": 0.04},
    )
    secondary_response = VerificationResponse(
        is_safe=False,
        needs_review=False,
        hallucination_rate=100.0,
        verification_rate=0.0,
        unverified_rate=0.0,
        claims=[secondary_claim],
        sanitized_text=rewritten_text,
    )

    fake_orch = FakeOrchestrator([secondary_response])
    pipeline = MitigationPipeline(
        rewriter=rewriter,
        orchestrator=fake_orch,
        fallback_mode=MitigationMode.BLOCK,
    )

    req = VerificationRequest(
        text=original_text,
        mitigation_mode=MitigationMode.REWRITE,
        fallback_mode=MitigationMode.BLOCK,
    )

    # When raise_on_block=True, must raise MitigationBlockedError
    with pytest.raises(MitigationBlockedError) as exc_info:
        pipeline.execute(req, initial_response, raise_on_block=True)
    assert exc_info.value.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Reset orchestrator for non-raising test
    fake_orch_2 = FakeOrchestrator([secondary_response])
    pipeline_2 = MitigationPipeline(
        rewriter=rewriter,
        orchestrator=fake_orch_2,
        fallback_mode=MitigationMode.BLOCK,
    )
    res_non_raise = pipeline_2.execute(req, initial_response, raise_on_block=False)
    assert res_non_raise.sanitized_text == DEFAULT_REFUSAL_MESSAGE


def test_nfr_safe_02_recursion_prevention_strictly_one_pass() -> None:
    """Verify NFR-SAFE-02: Secondary verification runs at most ONCE, preventing infinite hangs."""
    original_text = "Contradicted sentence."
    initial_claims = [
        ClaimReport(
            claim_id=0,
            claim_text="Contradicted sentence",
            original_sentence=original_text,
            state=ClaimState.CONTRADICTED,
            probabilities={"contradiction": 0.90, "entailment": 0.05, "neutral": 0.05},
        )
    ]
    initial_response = VerificationResponse(
        is_safe=False,
        needs_review=False,
        hallucination_rate=100.0,
        verification_rate=0.0,
        unverified_rate=0.0,
        claims=initial_claims,
        sanitized_text=original_text,
    )

    rewrite_call_count = 0

    class CountingGenerator(MockGenerator):
        def generate(
            self,
            prompt: str,
            max_new_tokens: int = 512,
            temperature: float = 0.0,
            **kwargs: Any,
        ) -> str:
            nonlocal rewrite_call_count
            rewrite_call_count += 1
            return "Still contradictory statement."

    generator = CountingGenerator()
    rewriter = TargetedRewriter(generator=generator)

    # Verifier repeatedly returns contradiction
    stub_contra = VerificationResponse(
        is_safe=False,
        needs_review=False,
        hallucination_rate=100.0,
        verification_rate=0.0,
        unverified_rate=0.0,
        claims=[
            ClaimReport(
                claim_id=0,
                claim_text="Still contradictory statement",
                original_sentence="Still contradictory statement.",
                state=ClaimState.CONTRADICTED,
                probabilities={"contradiction": 0.95, "entailment": 0.02, "neutral": 0.03},
            )
        ],
        sanitized_text="Still contradictory statement.",
    )
    fake_orch = FakeOrchestrator([stub_contra, stub_contra, stub_contra])

    pipeline = MitigationPipeline(
        rewriter=rewriter,
        orchestrator=fake_orch,
    )

    req = VerificationRequest(
        text=original_text,
        mitigation_mode=MitigationMode.REWRITE,
    )
    res = pipeline.execute(req, initial_response)

    # Strictly 1 rewrite invocation and 1 secondary verification pass!
    assert rewrite_call_count == 1
    assert fake_orch.call_count == 1
    assert DEFAULT_REDACTION_TOKEN in res.sanitized_text


def test_ut_mit_03_pipeline_clean_text_short_circuits() -> None:
    """Verify that text with zero contradictions does not trigger rewriter or secondary pass."""
    text = "Jupiter is the largest planet in our solar system."
    claims = [
        ClaimReport(
            claim_id=0,
            claim_text=text,
            state=ClaimState.SUPPORTED,
            probabilities={"entailment": 0.99, "neutral": 0.01, "contradiction": 0.0},
        )
    ]
    resp = VerificationResponse(
        is_safe=True,
        needs_review=False,
        hallucination_rate=0.0,
        verification_rate=100.0,
        unverified_rate=0.0,
        claims=claims,
        sanitized_text=text,
    )

    fake_orch = FakeOrchestrator([])
    mock_gen = MockGenerator()
    rewriter = TargetedRewriter(generator=mock_gen)
    pipeline = MitigationPipeline(rewriter=rewriter, orchestrator=fake_orch)

    req = VerificationRequest(text=text, mitigation_mode=MitigationMode.REWRITE)
    result = pipeline.execute(req, resp)

    assert result.sanitized_text == text
    assert fake_orch.call_count == 0


# ==============================================================================
# Slice 3: System & Gateway Integration Tests (ST_MIT_02 / ST_MIT_03 / REQ-MIT-03 / REQ-MIT-04)
# ==============================================================================


def test_st_mit_02_e2e_orchestrator_successful_rewrite_and_secondary_verification() -> None:
    """ST_MIT_02: End-to-end WaterfallOrchestrator rewrite with approved secondary verification."""
    # Setup mock verifier and mock generator
    mock_nli = MockNLIVerifier()
    mock_gen = MockGenerator()
    rewriter = TargetedRewriter(generator=mock_gen)

    # Context document establishes grounding truth: Paris is the capital of France
    context_doc = "The capital of France is Paris."
    hallucinated_text = "The capital of France is Berlin."

    from src.retrieval.store import DualIndexStore

    in_mem_store = DualIndexStore(in_memory=True)
    try:
        orchestrator = WaterfallOrchestrator(
            store=in_mem_store,
            nli_verifier=mock_nli,
            rewriter=rewriter,
            enable_mitigation=True,
        )

        req = VerificationRequest(
            text=hallucinated_text,
            context_documents=[context_doc],
            max_retrieval_tier=1,
            mitigation_mode=MitigationMode.REWRITE,
        )

        response = orchestrator.verify(req)

        # Initial text was contradicted (Berlin vs Paris), rewriter corrected it,
        # and secondary verification verified Paris as SUPPORTED.
        assert response.is_safe is True
        assert "Paris" in response.sanitized_text
        assert "Berlin" not in response.sanitized_text
        assert response.verification_rate == 100.0
        assert response.hallucination_rate == 0.0
    finally:
        in_mem_store.close()


def test_st_mit_02_e2e_orchestrator_fallback_cascade_to_redact() -> None:
    """ST_MIT_02: Secondary verification detects residual contradiction and cascades to REDACT."""
    mock_nli = MockNLIVerifier()

    # Generator rewrites to another contradiction: Berlin -> still Berlin (or ungrounded)
    class DefectiveGenerator(MockGenerator):
        def generate(
            self,
            prompt: str,
            max_new_tokens: int = 512,
            temperature: float = 0.0,
            **kwargs: Any,
        ) -> str:
            return "The capital of France is Berlin."

    rewriter = TargetedRewriter(generator=DefectiveGenerator())
    context_doc = "The capital of France is Paris."
    hallucinated_text = "The capital of France is Berlin."

    from src.retrieval.store import DualIndexStore

    in_mem_store = DualIndexStore(in_memory=True)
    try:
        orchestrator = WaterfallOrchestrator(
            store=in_mem_store,
            nli_verifier=mock_nli,
            rewriter=rewriter,
            enable_mitigation=True,
        )

        req = VerificationRequest(
            text=hallucinated_text,
            context_documents=[context_doc],
            max_retrieval_tier=1,
            mitigation_mode=MitigationMode.REWRITE,
            fallback_mode=MitigationMode.REDACT,
        )

        response = orchestrator.verify(req)

        # Residual contradiction causes cascade to redact
        assert DEFAULT_REDACTION_TOKEN in response.sanitized_text
        assert "Berlin" not in response.sanitized_text
    finally:
        in_mem_store.close()


def test_st_mit_03_e2e_gateway_verify_endpoint_rewrite_approval() -> None:
    """ST_MIT_03: POST /verify endpoint executes rewrite and approves secondary pass."""
    from fastapi.testclient import TestClient
    from src.gateway.app import create_app
    from src.gateway import router as gateway_router
    from src.retrieval.store import DualIndexStore

    mock_nli = MockNLIVerifier()
    mock_gen = MockGenerator()
    rewriter = TargetedRewriter(generator=mock_gen)
    in_mem_store = DualIndexStore(in_memory=True)

    custom_orch = WaterfallOrchestrator(
        store=in_mem_store,
        nli_verifier=mock_nli,
        rewriter=rewriter,
        enable_mitigation=True,
    )

    # Monkeypatch gateway default orchestrator and store for isolation
    original_orch = gateway_router._default_orchestrator
    original_store = gateway_router._default_store
    gateway_router._default_store = in_mem_store
    gateway_router._default_orchestrator = custom_orch

    try:
        app = create_app()
        with TestClient(app) as client:
            payload = {
                "text": "The capital of France is Berlin.",
                "context_documents": ["The capital of France is Paris."],
                "max_retrieval_tier": 1,
                "mitigation_mode": "rewrite",
            }

            res = client.post("/verify", json=payload)
            assert res.status_code == 200
            data = res.json()
            assert data["is_safe"] is True
            assert "Paris" in data["sanitized_text"]
            assert "Berlin" not in data["sanitized_text"]
    finally:
        gateway_router._default_orchestrator = original_orch
        gateway_router._default_store = original_store
        in_mem_store.close()


def test_st_mit_03_e2e_gateway_verify_endpoint_block_mode_raises_422() -> None:
    """ST_MIT_03: Verify POST /verify endpoint in BLOCK mode returns HTTP 422 on contradiction."""
    from fastapi.testclient import TestClient
    from src.gateway.app import create_app
    from src.gateway import router as gateway_router
    from src.retrieval.store import DualIndexStore

    mock_nli = MockNLIVerifier()
    in_mem_store = DualIndexStore(in_memory=True)
    custom_orch = WaterfallOrchestrator(
        store=in_mem_store,
        nli_verifier=mock_nli,
        enable_mitigation=True,
        raise_on_block=True,
    )

    original_orch = gateway_router._default_orchestrator
    original_store = gateway_router._default_store
    gateway_router._default_store = in_mem_store
    gateway_router._default_orchestrator = custom_orch

    try:
        app = create_app()
        with TestClient(app) as client:
            payload = {
                "text": "The capital of France is Berlin.",
                "context_documents": ["The capital of France is Paris."],
                "max_retrieval_tier": 1,
                "mitigation_mode": "block",
            }

            res = client.post("/verify", json=payload)
            assert res.status_code == 422
            data = res.json()
            assert DEFAULT_REFUSAL_MESSAGE in data.get("detail", "")
    finally:
        gateway_router._default_orchestrator = original_orch
        gateway_router._default_store = original_store
        in_mem_store.close()


def test_st_mit_03_e2e_gateway_verify_endpoint_rewrite_fallback_to_block_raises_422() -> None:
    """ST_MIT_03: Verify POST /verify in REWRITE mode with fallback_mode=BLOCK returns HTTP 422."""
    from fastapi.testclient import TestClient
    from src.gateway.app import create_app
    from src.gateway import router as gateway_router
    from src.retrieval.store import DualIndexStore

    mock_nli = MockNLIVerifier()

    class DefectiveGenerator(MockGenerator):
        def generate(
            self,
            prompt: str,
            max_new_tokens: int = 512,
            temperature: float = 0.0,
            **kwargs: Any,
        ) -> str:
            return "The capital of France is Berlin."

    rewriter = TargetedRewriter(generator=DefectiveGenerator())
    in_mem_store = DualIndexStore(in_memory=True)
    custom_orch = WaterfallOrchestrator(
        store=in_mem_store,
        nli_verifier=mock_nli,
        rewriter=rewriter,
        enable_mitigation=True,
        raise_on_block=True,
    )

    original_orch = gateway_router._default_orchestrator
    original_store = gateway_router._default_store
    gateway_router._default_store = in_mem_store
    gateway_router._default_orchestrator = custom_orch

    try:
        app = create_app()
        with TestClient(app) as client:
            payload = {
                "text": "The capital of France is Berlin.",
                "context_documents": ["The capital of France is Paris."],
                "max_retrieval_tier": 1,
                "mitigation_mode": "rewrite",
                "fallback_mode": "block",
            }

            res = client.post("/verify", json=payload)
            assert res.status_code == 422
            data = res.json()
            assert DEFAULT_REFUSAL_MESSAGE in data.get("detail", "")
    finally:
        gateway_router._default_orchestrator = original_orch
        gateway_router._default_store = original_store
        in_mem_store.close()


def test_st_mit_04_direct_redact_mode_through_orchestrator() -> None:
    """ST_MIT_04: Verify direct REDACT mode masks contradicted sentences via orchestrator."""
    mock_nli = MockNLIVerifier()
    from src.retrieval.store import DualIndexStore

    in_mem_store = DualIndexStore(in_memory=True)
    try:
        orchestrator = WaterfallOrchestrator(
            store=in_mem_store,
            nli_verifier=mock_nli,
            enable_mitigation=True,
        )

        req = VerificationRequest(
            text="The capital of France is Berlin. Water boils at 100 degrees.",
            context_documents=[
                "The capital of France is Paris.",
                "Water boils at 100 degrees Celsius.",
            ],
            max_retrieval_tier=1,
            mitigation_mode=MitigationMode.REDACT,
        )

        res = orchestrator.verify(req)
        assert DEFAULT_REDACTION_TOKEN in res.sanitized_text
        assert "Berlin" not in res.sanitized_text
        assert "Water boils at 100 degrees" in res.sanitized_text
    finally:
        in_mem_store.close()
