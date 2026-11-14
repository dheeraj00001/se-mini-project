"""Unit tests for the Core Gateway scaffolding and verification endpoint.

Covers:
- UT_GW_01: Application initialization, health check, CORS, and proxy stub (REQ-GW-01).
- UT_GW_02: POST /verify endpoint and Pydantic V2 schema validation (REQ-GW-02).
"""

from typing import Generator
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.gateway.app import create_app
from src.main import app as main_app
from src.schemas.verification import (
    ClaimReport,
    ClaimSegment,
    ClaimState,
    EvidenceChunk,
    MitigationMode,
    VerificationRequest,
    VerificationResponse,
)


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Fixture providing a reusable FastAPI test client."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


# ============================================================================
# UT_GW_01: Gateway Application, Health Check, CORS & Scaffolding
# ============================================================================


def test_ut_gw_01_health_check_endpoint(client: TestClient) -> None:
    """UT_GW_01: Verify GET /health responds with 200 and version metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "1.0.0"}
    assert "X-Process-Time-Ms" in response.headers


def test_ut_gw_01_cors_and_logging_middleware(client: TestClient) -> None:
    """UT_GW_01: Verify CORS wildcard headers and logging timing headers."""
    response = client.get("/health", headers={"Origin": "http://client.example.com"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "*"
    assert "X-Process-Time-Ms" in response.headers


def test_ut_gw_01_main_app_instance() -> None:
    """UT_GW_01: Verify src.main:app executes as a valid FastAPI instance."""
    with TestClient(main_app) as main_client:
        response = main_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "version": "1.0.0"}


def test_ut_gw_01_chat_completions_proxy_stub(client: TestClient) -> None:
    """UT_GW_01: Verify POST /v1/chat/completions proxy stub responds cleanly."""
    payload = {
        "model": "gemma-4b",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Dehallucination proxy initialized."


def test_ut_gw_01_global_exception_handler() -> None:
    """UT_GW_01: Verify global exception handler catches unhandled errors with 500."""
    app = create_app()

    @app.get("/trigger-error")
    def _error_route() -> None:
        raise RuntimeError("Unexpected failure")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/trigger-error")
        assert response.status_code == 500
        assert response.json()["error"] == "Internal Server Error"


# ============================================================================
# UT_GW_02: Verification Endpoint & Strict Schema Validation
# ============================================================================


def test_ut_gw_02_verification_request_defaults_and_immutability() -> None:
    """UT_GW_02: Verify schema defaults and ensure text is not mutated."""
    text_content = "  The sky is blue.  "
    req = VerificationRequest(text=text_content)
    assert req.text == text_content  # text is validated without in-place mutation
    assert req.prompt == ""
    assert req.context_documents == []
    assert req.max_retrieval_tier == 2
    assert req.mitigation_mode == MitigationMode.FLAG


def test_ut_gw_02_verification_request_invalid_tier() -> None:
    """UT_GW_02: Verify rejection of retrieval tiers outside 1..3."""
    with pytest.raises(ValidationError):
        VerificationRequest(text="Valid text", max_retrieval_tier=4)

    with pytest.raises(ValidationError):
        VerificationRequest(text="Valid text", max_retrieval_tier=0)


def test_ut_gw_02_verification_request_empty_text() -> None:
    """UT_GW_02: Verify rejection of empty or whitespace-only text."""
    with pytest.raises(ValidationError):
        VerificationRequest(text="")

    with pytest.raises(ValidationError):
        VerificationRequest(text="   ")


def test_ut_gw_02_verification_response_serialization() -> None:
    """UT_GW_02: Verify response schema serialization with mandatory claims."""
    claim = ClaimReport(
        claim_id=1,
        claim_text="The sky is blue.",
        state=ClaimState.SUPPORTED,
        probabilities={"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02},
        evidence_tier=1,
        evidence_text="The sky appears blue due to Rayleigh scattering.",
    )
    res = VerificationResponse(
        is_safe=True,
        needs_review=False,
        hallucination_rate=0.0,
        verification_rate=100.0,
        unverified_rate=0.0,
        claims=[claim],
        sanitized_text="The sky is blue.",
    )
    data = res.model_dump()
    assert data["is_safe"] is True
    assert data["claims"][0]["state"] == "SUPPORTED"


def test_ut_gw_02_claim_segment_contract() -> None:
    """UT_GW_02: Verify Student 2 ClaimSegment schema contract."""
    segment = ClaimSegment(
        claim_id=0,
        claim_text="The Eiffel Tower is in Paris.",
        original_sentence="I think the Eiffel Tower is in Paris.",
        has_epistemic_hedge=True,
    )
    assert segment.claim_id == 0
    assert segment.claim_text == "The Eiffel Tower is in Paris."
    assert segment.has_epistemic_hedge is True

    with pytest.raises(ValidationError):
        ClaimSegment(claim_id=-1, claim_text="Invalid", original_sentence="Invalid")


def test_ut_gw_02_evidence_chunk_contract() -> None:
    """UT_GW_02: Verify Student 3 EvidenceChunk schema contract."""
    chunk = EvidenceChunk(
        chunk_id="a1b2c3d4e5f678901234567890abcdef1234567890abcdef1234567890abcdef",
        text="Paris is the capital of France.",
        source_doc="geography_handbook.pdf",
        tier=2,
        score=0.88,
        metadata={"author": "PES"},
    )
    assert chunk.tier == 2
    assert chunk.score == 0.88
    assert chunk.metadata["author"] == "PES"

    with pytest.raises(ValidationError):
        EvidenceChunk(
            chunk_id="bad",
            text="text",
            source_doc="doc",
            tier=4,  # tier must be <= 3
        )


def test_ut_gw_02_verify_endpoint_success_stub(client: TestClient) -> None:
    """UT_GW_02: Verify POST /verify returns schema-compliant response stub."""
    payload = {"text": "The Eiffel Tower is in Paris."}
    response = client.post("/verify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_safe"] is True
    assert data["needs_review"] is False
    assert data["hallucination_rate"] == 0.0
    assert data["verification_rate"] == 0.0
    assert data["unverified_rate"] == 0.0
    assert data["claims"] == []
    assert data["sanitized_text"] == "The Eiffel Tower is in Paris."


def test_ut_gw_02_verify_endpoint_with_all_fields(client: TestClient) -> None:
    """UT_GW_02: Verify POST /verify with full optional parameter set."""
    payload = {
        "text": "Sample text to verify.",
        "prompt": "What is this?",
        "context_documents": ["Reference doc 1"],
        "max_retrieval_tier": 3,
        "mitigation_mode": "rewrite",
    }
    response = client.post("/verify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sanitized_text"] == "Sample text to verify."


def test_ut_gw_02_verify_endpoint_out_of_range_tier(client: TestClient) -> None:
    """UT_GW_02: Verify HTTP 422 on max_retrieval_tier > 3 or < 1."""
    res_high = client.post("/verify", json={"text": "Test", "max_retrieval_tier": 4})
    assert res_high.status_code == 422
    assert "error" in res_high.json()

    res_low = client.post("/verify", json={"text": "Test", "max_retrieval_tier": 0})
    assert res_low.status_code == 422
    assert "error" in res_low.json()


def test_ut_gw_02_verify_endpoint_missing_and_blank_text(client: TestClient) -> None:
    """UT_GW_02: Verify HTTP 422 on missing or empty text."""
    res_missing = client.post("/verify", json={})
    assert res_missing.status_code == 422
    assert "error" in res_missing.json()

    res_blank = client.post("/verify", json={"text": "   "})
    assert res_blank.status_code == 422
    assert "error" in res_blank.json()


def test_ut_gw_02_verify_endpoint_invalid_mitigation_mode(client: TestClient) -> None:
    """UT_GW_02: Verify HTTP 422 on invalid mitigation_mode."""
    res = client.post("/verify", json={"text": "Test", "mitigation_mode": "invalid_mode"})
    assert res.status_code == 422
    assert "error" in res.json()


def test_ut_gw_02_verify_endpoint_malformed_json(client: TestClient) -> None:
    """UT_GW_02: Verify HTTP 422 on malformed JSON payload."""
    res = client.post(
        "/verify",
        content="not a valid json string {",
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 422
    assert "error" in res.json()
