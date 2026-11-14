# 08 — Secondary Verification Loop & Mitigation Fallback Cascade

**What to build:**
A guardrail wrapper around rewritten responses that executes a mandatory single secondary verification pass, confirming that the rewriter introduced no new hallucinations; if secondary contradictions or `is_safe == false` persist, it automatically cascades to `redact` (sentence masking) or `block` (refusal response).

**Sprint:** Sprint 2  
**Owner:** Student 4 (Mitigation)  
**Story Points:** 5 SP  
**SRS Reference:** REQ-MIT-03, REQ-MIT-04, NFR-SAFE-01, NFR-SAFE-02, ADR-0006  
**Blocked by:** 07 — Quantized Local Gemma GPU Generator & Targeted Rewriter  
**Status:** ready-for-agent  

### Context & Scope
Generative rewrites can introduce secondary hallucinations. To eliminate this risk without entering infinite loops, all rewritten responses undergo a single secondary verification pass (`NFR-SAFE-02`). If the rewritten response still contains contradictions ($P \ge 0.70$) or trips the safety gate ($P \ge 0.85$), the system immediately halts rewriting and cascades to deterministic redaction (*"[Claim redacted: Disproven by reference knowledge]"*) or an HTTP refusal message.

### Acceptance Criteria
- [ ] Rewritten text is submitted to a secondary verification pass using the pipeline from Ticket 06.
- [ ] If secondary verification detects zero contradictions and `is_safe == true`, the rewritten response is approved and returned.
- [ ] If secondary contradictions persist, the loop terminates immediately (strictly 1 retry maximum to satisfy `NFR-SAFE-02`).
- [ ] In `redact` mode (or fallback), contradicted sentences are replaced with standardized redaction tokens: `"[Claim redacted: Disproven by reference knowledge]"`.
- [ ] In `block` mode, an HTTP 422 error or standardized refusal message (*"Unable to provide a factually grounded response."*) is returned.
- [ ] Integration tests (`tests/test_secondary_verification.py`) simulate both successful rewrites and forced fallback cascades (`UT_MIT_03`, `UT_MIT_04`, `ST_MIT_02`).

### Definition of Done
Pipeline demonstrates automated recovery from hallucination: rewriting clean text, verifying it, or falling back safely to redaction/blocking without hanging or crashing.
