"""End-to-end integration tests for NLI-Aware Waterfall Orchestration & Sprint-1 MVP.

Implements and validates:
- ST_E2E_02: End-to-end verification via POST /verify and pipeline orchestrator.
- ST_RET_02: Sequential Waterfall escalation (Tier 1 -> Tier 2 -> Unverified).
- REQ-GW-02, REQ-RET-05, REQ-VER-05, REQ-VER-06, ADR-0004, ADR-0005.
- NFR-PERF-04: End-to-end latency <= 500ms for 5 claims.
"""

from src.schemas.verification import (
    ClaimState,
    VerificationRequest,
    VerificationResponse,
)
from src.verifier.orchestrator import WaterfallOrchestrator, get_orchestrator


def test_ut_orch_01_orchestrator_initialization_and_empty_text() -> None:
    """Slice 1: Verify orchestrator initializes and handles empty or non-factual text."""
    orchestrator = get_orchestrator(use_mock=True)
    assert isinstance(orchestrator, WaterfallOrchestrator)

    # Conversational non-factual text with no extractable atomic claims
    req = VerificationRequest(text="Hello there! How are you doing today?")
    resp = orchestrator.verify(req)

    assert isinstance(resp, VerificationResponse)
    assert resp.is_safe is True
    assert resp.needs_review is False
    assert resp.hallucination_rate == 0.0
    assert resp.verification_rate == 0.0
    assert resp.unverified_rate == 0.0
    assert resp.claims == []
    assert resp.sanitized_text == req.text


def test_ut_orch_02_tier1_short_circuit_on_decisive_evidence() -> None:
    """Slice 2: Verify decisive Tier 1 evidence short-circuits and prevents querying Tier 2."""
    from unittest.mock import MagicMock
    from src.retrieval.adapters import Tier1ContextAdapter, Tier2HybridAdapter, RetrievalAdapter

    tier1_adapter = Tier1ContextAdapter()
    mock_tier2_adapter = MagicMock(spec=Tier2HybridAdapter)

    retrieval_adapter = RetrievalAdapter(
        tier1_adapter=tier1_adapter,
        tier2_adapter=mock_tier2_adapter,
    )

    orchestrator = get_orchestrator(
        retrieval_adapter=retrieval_adapter,
        use_mock=True,
    )

    req = VerificationRequest(
        text="The Eiffel Tower is in Paris.",
        context_documents=["The Eiffel Tower is located in Paris, France."],
        max_retrieval_tier=2,
    )

    resp = orchestrator.verify(req)

    assert len(resp.claims) == 1
    claim = resp.claims[0]
    assert claim.state == ClaimState.SUPPORTED
    assert claim.evidence_tier == 1
    assert claim.evidence_text is not None
    assert "Paris" in claim.evidence_text
    assert claim.probabilities["entailment"] >= 0.70

    # Short-circuit check: Tier 2 must never be called!
    assert mock_tier2_adapter.search.call_count == 0


def test_ut_orch_03_tier1_inconclusive_escalates_to_tier2() -> None:
    """Slice 3: Verify inconclusive or missing Tier 1 evidence escalates to Tier 2."""
    from src.retrieval.embedder import MockEmbedder
    from src.retrieval.store import DualIndexStore
    from src.schemas.verification import EvidenceChunk

    store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))
    chunk_kb = EvidenceChunk(
        chunk_id="kb_chunk_1",
        text="Apollo 11 landed humans on the Moon on July 20, 1969.",
        source_doc="space_history.txt",
        tier=2,
    )
    store.add_chunks([chunk_kb])

    orchestrator = get_orchestrator(store=store, use_mock=True)

    # Inconclusive Tier 1 context: "Space missions are exciting."
    req = VerificationRequest(
        text="Apollo 11 landed humans on the Moon.",
        context_documents=["Space missions are exciting and require teamwork."],
        max_retrieval_tier=2,
    )

    resp = orchestrator.verify(req)

    assert len(resp.claims) == 1
    claim = resp.claims[0]
    assert claim.state == ClaimState.SUPPORTED
    assert claim.evidence_tier == 2
    assert claim.evidence_text == chunk_kb.text
    assert claim.probabilities["entailment"] >= 0.70


def test_ut_orch_04_unverified_claim_fallback_and_null_evidence() -> None:
    """Slice 4: Verify claim with no decisive evidence across tiers is UNVERIFIED with null."""
    from src.retrieval.embedder import MockEmbedder
    from src.retrieval.store import DualIndexStore

    # Empty knowledge base and empty in-context documents
    store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))
    orchestrator = get_orchestrator(store=store, use_mock=True)

    req = VerificationRequest(
        text="The mythical city of Atlantis had advanced flying machines.",
        context_documents=[],
        max_retrieval_tier=2,
    )

    resp = orchestrator.verify(req)

    assert len(resp.claims) == 1
    claim = resp.claims[0]
    assert claim.state == ClaimState.UNVERIFIED
    assert claim.evidence_tier is None
    assert claim.evidence_text is None
    assert claim.probabilities["neutral"] >= 0.50

    # Document-level checks:
    # Never classify unverified as hallucination (hallucination_rate must be 0)
    assert resp.hallucination_rate == 0.0
    assert resp.verification_rate == 0.0
    assert resp.unverified_rate == 100.0
    assert resp.is_safe is True
    assert resp.needs_review is True


def test_ut_orch_05_document_reliability_rates_and_safety_gate() -> None:
    """Slice 5: Multi-claim document verifying document rates and dual-flag safety gating."""
    import math
    from src.retrieval.embedder import MockEmbedder
    from src.retrieval.store import DualIndexStore
    from src.schemas.verification import EvidenceChunk

    store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))
    # Contradiction premise: "The capital of Germany is Berlin, France's capital is Paris."
    c_paris = EvidenceChunk(
        chunk_id="kb_paris",
        text="Paris is the capital of France. Berlin is the capital of Germany.",
        source_doc="capitals.txt",
        tier=2,
    )
    store.add_chunks([c_paris])

    orchestrator = get_orchestrator(store=store, use_mock=True)

    # 3 claims in text:
    # 1. "The sky is blue." -> supported by Tier 1 context doc
    # 2. "The capital of France is Berlin." -> contradicted by Tier 2 KB
    # 3. "Dolphins have purple spots." -> unverified (no evidence anywhere)
    text = "The sky is blue. " "The capital of France is Berlin. " "Dolphins have purple spots."
    req = VerificationRequest(
        text=text,
        context_documents=["The sky is blue due to Rayleigh scattering."],
        max_retrieval_tier=2,
    )

    resp = orchestrator.verify(req)

    assert len(resp.claims) == 3
    # Check claim states
    c1, c2, c3 = resp.claims[0], resp.claims[1], resp.claims[2]

    assert c1.state == ClaimState.SUPPORTED
    assert c1.evidence_tier == 1

    assert c2.state == ClaimState.CONTRADICTED
    assert c2.evidence_tier == 2
    assert c2.probabilities["contradiction"] >= 0.85

    assert c3.state == ClaimState.UNVERIFIED
    assert c3.evidence_tier is None
    assert c3.evidence_text is None

    # Document-level rates: 1 out of 3 each = 33.33%
    assert math.isclose(resp.hallucination_rate, 100.0 / 3.0, rel_tol=1e-3)
    assert math.isclose(resp.verification_rate, 100.0 / 3.0, rel_tol=1e-3)
    assert math.isclose(resp.unverified_rate, 100.0 / 3.0, rel_tol=1e-3)

    # Safety Gating:
    # is_safe is False because c2 contradiction >= 0.85
    assert resp.is_safe is False
    # needs_review is True because c3 is UNVERIFIED
    assert resp.needs_review is True


def test_st_e2e_02_gateway_verify_endpoint_full_waterfall() -> None:
    """Slice 6: ST_E2E_02 end-to-end verification via HTTP POST /verify."""
    from fastapi.testclient import TestClient
    from src.gateway.app import create_app
    from src.gateway import router as gateway_router
    from src.retrieval.embedder import MockEmbedder
    from src.retrieval.store import DualIndexStore

    # Reset global default store & orchestrator with isolated in-memory store
    mock_store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))
    gateway_router._default_store = mock_store
    gateway_router._default_orchestrator = get_orchestrator(store=mock_store, use_mock=True)

    app = create_app()
    with TestClient(app) as client:
        # 1. Ingest knowledge into Tier 2 local KB via POST /knowledge/ingest
        ingest_payload = {
            "text": (
                "Apollo 11 landed humans on the Moon in July 1969. " "Neil Armstrong was commander."
            ),
            "source_doc": "space_history.txt",
        }
        ingest_res = client.post("/knowledge/ingest", json=ingest_payload)
        assert ingest_res.status_code == 201
        assert ingest_res.json()["chunks_indexed"] >= 1

        # 2. Verify text exercising both Tier 1 and Tier 2 resolution
        verify_payload = {
            "text": "The Eiffel Tower is in Paris. Apollo 11 landed on the Moon.",
            "context_documents": ["The Eiffel Tower is located in Paris, France."],
            "max_retrieval_tier": 2,
        }
        res = client.post("/verify", json=verify_payload)
        assert res.status_code == 200
        data = res.json()

        assert data["is_safe"] is True
        assert data["needs_review"] is False
        assert data["hallucination_rate"] == 0.0
        assert data["verification_rate"] == 100.0
        assert data["unverified_rate"] == 0.0
        assert len(data["claims"]) == 2

        # Claim 1: Tier 1 match
        c1 = data["claims"][0]
        assert c1["state"] == "SUPPORTED"
        assert c1["evidence_tier"] == 1
        assert "Paris" in c1["evidence_text"]

        # Claim 2: Tier 2 match
        c2 = data["claims"][1]
        assert c2["state"] == "SUPPORTED"
        assert c2["evidence_tier"] == 2
        assert "Apollo 11" in c2["evidence_text"]


def test_nfr_perf_04_e2e_verification_latency_under_500ms() -> None:
    """Slice 7: NFR-PERF-04 end-to-end verification latency <= 500ms for 5 claims."""
    import time
    from src.retrieval.embedder import MockEmbedder
    from src.retrieval.store import DualIndexStore
    from src.schemas.verification import EvidenceChunk

    store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))
    # Pre-populate store with knowledge passages
    chunks = [
        EvidenceChunk(
            chunk_id=f"perf_chunk_{i}",
            text=f"Passage number {i} provides verified facts about scientific principles.",
            source_doc="science.txt",
            tier=2,
        )
        for i in range(10)
    ]
    store.add_chunks(chunks)

    orchestrator = get_orchestrator(store=store, use_mock=True)

    text_5_claims = (
        "The Earth revolves around the Sun. "
        "Water freezes at zero degrees Celsius. "
        "Light travels faster than sound. "
        "DNA carries genetic instructions. "
        "Oxygen is essential for human respiration."
    )
    context_docs = [
        "The Earth revolves around the Sun once each year.",
        "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
        "Light travels at approximately 300,000 kilometers per second, faster than sound.",
    ]

    req = VerificationRequest(
        text=text_5_claims,
        context_documents=context_docs,
        max_retrieval_tier=2,
    )

    # Warm-up pass
    orchestrator.verify(req)

    # Timed measurement pass
    t0 = time.perf_counter()
    resp = orchestrator.verify(req)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(resp.claims) == 5
    assert (
        elapsed_ms <= 500.0
    ), f"E2E verification latency {elapsed_ms:.2f}ms exceeded 500ms requirement (NFR-PERF-04)"


def test_st_ret_02_tier_ceiling_enforcement_max_tier_1() -> None:
    """Slice 8: ST_RET_02 verify max_retrieval_tier=1 prevents Tier 2 escalation."""
    from unittest.mock import MagicMock
    from src.retrieval.adapters import Tier1ContextAdapter, Tier2HybridAdapter, RetrievalAdapter

    tier1_adapter = Tier1ContextAdapter()
    mock_tier2_adapter = MagicMock(spec=Tier2HybridAdapter)

    retrieval_adapter = RetrievalAdapter(
        tier1_adapter=tier1_adapter,
        tier2_adapter=mock_tier2_adapter,
    )

    orchestrator = get_orchestrator(
        retrieval_adapter=retrieval_adapter,
        use_mock=True,
    )

    # Inconclusive Tier 1 context and max_retrieval_tier=1
    req = VerificationRequest(
        text="Apollo 11 landed humans on the Moon.",
        context_documents=["Astronomy is a fascinating subject."],
        max_retrieval_tier=1,
    )

    resp = orchestrator.verify(req)

    assert len(resp.claims) == 1
    assert resp.claims[0].state == ClaimState.UNVERIFIED
    # Tier 2 must NEVER be called because max_retrieval_tier == 1
    assert mock_tier2_adapter.search.call_count == 0


def test_st_ret_02_epistemic_hedges_and_mixed_claims() -> None:
    """Slice 8: ST_RET_02 & ST_E2E_02 verify epistemic hedges and conversational filler."""
    from src.retrieval.embedder import MockEmbedder
    from src.retrieval.store import DualIndexStore
    from src.schemas.verification import EvidenceChunk

    store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))
    c_python = EvidenceChunk(
        chunk_id="kb_python",
        text="Python was created by Guido van Rossum and released in 1991.",
        source_doc="python_history.txt",
        tier=2,
    )
    store.add_chunks([c_python])

    orchestrator = get_orchestrator(store=store, use_mock=True)

    # Input contains greeting + hedged factual claim + sign-off filler
    text = (
        "Hello! Sure, I can help with that. "
        "I believe that Guido van Rossum created Python. "
        "Hope this helps! Thank you!"
    )
    req = VerificationRequest(
        text=text,
        max_retrieval_tier=2,
    )

    resp = orchestrator.verify(req)

    # Greetings and filler must be filtered out; only the hedged claim is evaluated
    assert len(resp.claims) == 1
    claim = resp.claims[0]
    assert "Guido van Rossum created Python" in claim.claim_text
    assert claim.state == ClaimState.SUPPORTED
    assert claim.evidence_tier == 2
    assert resp.is_safe is True
    assert resp.needs_review is False
    assert resp.verification_rate == 100.0


def test_ut_orch_properties_and_accessors() -> None:
    """Slice 8: Verify orchestrator property accessors and component wiring."""
    orchestrator = get_orchestrator(use_mock=True)
    assert orchestrator.store is not None
    assert orchestrator.retrieval_adapter is not None
    assert orchestrator.nli_verifier is not None
    assert orchestrator.segmenter is not None


def test_retrieval_waterfall_module_reexport() -> None:
    """Verify src.retrieval.waterfall re-exports WaterfallOrchestrator (REQ-RET-05)."""
    import pytest
    import src.retrieval.waterfall as wf

    assert wf.WaterfallOrchestrator is WaterfallOrchestrator
    assert wf.VerificationOrchestrator is WaterfallOrchestrator
    assert wf.get_orchestrator is get_orchestrator

    with pytest.raises(AttributeError):
        getattr(wf, "non_existent_attribute")


def test_orchestrator_custom_injected_components() -> None:
    """Verify WaterfallOrchestrator accepts custom injected verifier and adapter without store."""
    from src.retrieval.adapters import get_retrieval_adapter
    from src.verifier.nli import MockNLIVerifier

    mock_nli = MockNLIVerifier()
    adapter = get_retrieval_adapter()
    orchestrator = get_orchestrator(nli_verifier=mock_nli, retrieval_adapter=adapter)
    assert orchestrator.nli_verifier is mock_nli
    assert orchestrator.retrieval_adapter is adapter
    assert orchestrator.store is adapter.tier2_adapter.store

    # Test direct WaterfallOrchestrator init branches with and without adapter
    direct_orch = WaterfallOrchestrator(retrieval_adapter=adapter)
    assert direct_orch.store is adapter.tier2_adapter.store

    direct_default = WaterfallOrchestrator()
    assert direct_default.store is not None


# ==============================================================================
# Student 2 Validation Suite: Document Reliability Telemetry (REQ-VER-05)
# ==============================================================================


def test_student_2_telemetry_summary_generation() -> None:
    """Verify Student 2 get_verification_telemetry_summary accurately compiles rates."""
    from src.schemas.verification import ClaimReport, ClaimSegment, ClaimState, VerificationResponse
    from src.verifier.orchestrator import get_verification_telemetry_summary

    claims = [
        ClaimReport(
            claim=ClaimSegment(claim_id="c1", claim_text="Paris is the capital of France.", order=0),
            state=ClaimState.SUPPORTED,
            confidence=0.98,
            probabilities={"entailment": 0.98, "neutral": 0.01, "contradiction": 0.01},
        ),
        ClaimReport(
            claim=ClaimSegment(claim_id="c2", claim_text="Berlin is in Spain.", order=1),
            state=ClaimState.CONTRADICTED,
            confidence=0.92,
            probabilities={"entailment": 0.01, "neutral": 0.07, "contradiction": 0.92},
        ),
        ClaimReport(
            claim=ClaimSegment(claim_id="c3", claim_text="There are 500 stars in this cluster.", order=2),
            state=ClaimState.UNVERIFIED,
            confidence=0.40,
            probabilities={"entailment": 0.20, "neutral": 0.70, "contradiction": 0.10},
        ),
    ]

    response = VerificationResponse(
        original_text="Test paragraph",
        verified_text="Test paragraph",
        claims=claims,
        is_safe=False,
        needs_review=True,
        hallucination_rate=33.33,
        verification_rate=33.33,
        unverified_rate=33.33,
    )

    summary = get_verification_telemetry_summary(response)
    assert summary["total_claims"] == 3
    assert summary["supported_count"] == 1
    assert summary["contradicted_count"] == 1
    assert summary["unverified_count"] == 1
    assert summary["is_safe"] is False
    assert summary["needs_review"] is True
    assert summary["hallucination_rate"] == 33.33

