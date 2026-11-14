# 01 — Core Gateway Scaffolding, Verification Endpoint & CI Pipeline

**What to build:**
A running FastAPI service exposing the foundational `POST /verify` endpoint and health checks, with strict Pydantic schemas (`VerificationRequest`, `VerificationResponse`), CORS, and an automated GitHub Actions CI pipeline running linting, formatting, and unit tests on every pull request from day one.

**Sprint:** Sprint 1  
**Owner:** Student 1 (Gateway)  
**Story Points:** 3 SP  
**SRS Reference:** REQ-GW-01, REQ-GW-02, NFR-QUAL-01, NFR-QUAL-02  
**Blocked by:** None — can start immediately  
**Status:** completed  

### Context & Scope
This is the front door of the system and the foundational repository scaffold. As mandated by PES University SE guidelines, CI/CD automation must be established in Sprint 1. This ticket sets up the FastAPI application factory, logging middleware, error handling, input validation, and GitHub Actions workflow (`.github/workflows/ci.yml`).

### Acceptance Criteria
- [x] FastAPI application initializes cleanly and responds to `GET /health` with `{"status": "healthy", "version": "1.0.0"}`.
- [x] Endpoint `POST /verify` accepts `VerificationRequest` (text, prompt, context_documents, max_retrieval_tier, mitigation_mode) and returns schema-compliant `VerificationResponse` stub.
- [x] Strict Pydantic V2 validation rejects malformed JSON and out-of-range parameters (`max_retrieval_tier` > 3) with HTTP 422.
- [x] GitHub Actions workflow `.github/workflows/ci.yml` triggers on PRs to `main`, executing `flake8`, `mypy`, and `pytest`.
- [x] Unit tests (`tests/test_gateway_scaffold.py`) pass with 100% route coverage (`UT_GW_01`, `UT_GW_02`).

### Definition of Done
FastAPI server runs locally on port 8000, passes local `pytest`, and successfully triggers a green GitHub Actions check on a test branch/PR.
