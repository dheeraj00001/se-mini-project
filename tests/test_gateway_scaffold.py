import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.gateway.app import create_app
from src.main import app as main_app
from src.schemas.verification import (
    ClaimReport,
    ClaimState,
    MitigationMode,
    VerificationRequest,
    VerificationResponse,
)


def test_verification_request_defaults_and_validation() -> None:
    req = VerificationRequest(text="The sky is blue.")
    assert req.text == "The sky is blue."
    assert req.prompt == ""
    assert req.context_documents == []
    assert req.max_retrieval_tier == 2
    assert req.mitigation_mode == MitigationMode.FLAG


def test_verification_request_invalid_tier() -> None:
    with pytest.raises(ValidationError):
        VerificationRequest(text="Valid text", max_retrieval_tier=4)

    with pytest.raises(ValidationError):
        VerificationRequest(text="Valid text", max_retrieval_tier=0)


def test_verification_request_empty_text() -> None:
    with pytest.raises(ValidationError):
        VerificationRequest(text="")

    with pytest.raises(ValidationError):
        VerificationRequest(text="   ")


def test_verification_response_serialization() -> None:
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


def test_health_check_endpoint() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "1.0.0"}


def test_verify_endpoint_success_stub() -> None:
    app = create_app()
    client = TestClient(app)
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


def test_verify_endpoint_with_all_fields() -> None:
    app = create_app()
    client = TestClient(app)
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


def test_verify_endpoint_out_of_range_tier() -> None:
    app = create_app()
    client = TestClient(app)
    # Tier > 3
    res_high = client.post("/verify", json={"text": "Test", "max_retrieval_tier": 4})
    assert res_high.status_code == 422

    # Tier < 1
    res_low = client.post("/verify", json={"text": "Test", "max_retrieval_tier": 0})
    assert res_low.status_code == 422


def test_verify_endpoint_missing_and_blank_text() -> None:
    app = create_app()
    client = TestClient(app)
    # Missing text
    res_missing = client.post("/verify", json={})
    assert res_missing.status_code == 422

    # Blank text
    res_blank = client.post("/verify", json={"text": "   "})
    assert res_blank.status_code == 422


def test_verify_endpoint_invalid_mitigation_mode() -> None:
    app = create_app()
    client = TestClient(app)
    res = client.post("/verify", json={"text": "Test", "mitigation_mode": "invalid_mode"})
    assert res.status_code == 422


def test_verify_endpoint_malformed_json() -> None:
    app = create_app()
    client = TestClient(app)
    res = client.post(
        "/verify",
        content="not a valid json string {",
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 422


def test_main_app_and_cors_middleware() -> None:
    client = TestClient(main_app)
    # Test GET through main_app with Origin header
    response = client.get("/health", headers={"Origin": "http://example.com"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://example.com"
    assert response.headers.get("access-control-allow-credentials") == "true"
