# 06 — NLI-Aware Waterfall Orchestration & Sprint-1 MVP Integration

**What to build:**
The central verification orchestrator that ties Claim Segmentation, Retrieval Adapters, and NLI Verification together into an NLI-aware cascading Waterfall loop, exposing the end-to-end working pipeline through `POST /verify` to deliver the working MVP for the Sprint-1 demo video.

**Sprint:** Sprint 1  
**Owner:** Student 2 (Balaji - @avulabalaji35-rgb | Verification Lead) & Student 1  
**Story Points:** 5 SP  
**SRS Reference:** REQ-GW-02, REQ-RET-05, REQ-VER-05, REQ-VER-06, ADR-0004  
**GitHub Issue:** #9  
**Blocked by:** 01 — Core Gateway Scaffolding, 05 — CPU-Based DeBERTa-v3 NLI Verification  
**Status:** done  

### Context & Scope
This ticket integrates all Sprint-1 vertical components into a working product:
1. Segments incoming text into atomic claims (Module 2).
2. For each claim, executes the NLI-aware Waterfall: queries Tier 1; runs NLI; short-circuits if decisive ($P \ge 0.70$). If inconclusive/neutral, queries Tier 2 (Qdrant+BM25); runs NLI. If still inconclusive, marks `UNVERIFIED`.
3. Compiles the complete `VerificationResponse` with claim breakdown and document rates.
4. *Sprint 1 Parallel Spike (Student 4)*: Benchmarks 4-bit Gemma model loading on RTX 2050 GPU to verify VRAM allocation $\le 2.5\text{ GB}$ before Sprint 2.

### Acceptance Criteria
- [x] Calling `POST /verify` with text and optional context documents returns a complete, accurate `VerificationResponse`.
- [x] Waterfall short-circuiting works correctly: Tier 1 matches do not query Tier 2.
- [x] Inconclusive Tier 1 claims automatically escalate to Tier 2 hybrid retrieval.
- [x] If no evidence is found in Tier 1 or Tier 2, the claim is classified as `UNVERIFIED` with `evidence_text = null` (not marked as contradicted).
- [x] End-to-end latency for a 5-claim text completes in $\le 500\text{ ms}$ on local hardware (`NFR-PERF-04`).
- [x] Integration tests (`tests/test_waterfall_e2e.py`) pass with 100% success (`ST_E2E_02`, `ST_RET_02`).
- [x] A 1-minute video demonstrating the working `/verify` endpoint on real text is recorded and staged for submission.

### Definition of Done
The system runs end-to-end via `uvicorn src.main:app`, successfully processes verification requests via Swagger UI/curl, and passes all integration tests in GitHub Actions CI.
