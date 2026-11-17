# 05 — CPU-Based DeBERTa-v3 NLI Verification & Reliability Scoring

**What to build:**
A CPU-optimized Natural Language Inference engine executing `cross-encoder/nli-deberta-v3-base` on premise-hypothesis pairs to produce normalized probabilities, assign 3-way discrete states (`SUPPORTED`, `CONTRADICTED`, `UNVERIFIED`), compute document reliability rates, and evaluate safety gating flags.

**Sprint:** Sprint 1  
**Owner:** Student 2 (Balaji - @avulabalaji35-rgb | NLP & Verification Lead)  
**Story Points:** 5 SP  
**SRS Reference:** REQ-VER-04, REQ-VER-05, REQ-VER-06, NFR-PERF-02, ADR-0002, ADR-0005  
**GitHub Issue:** #8  
**Blocked by:** 02 — Deterministic Sentence & Clause Claim Segmenter, 04 — Hybrid Retrieval Adapters  
**Status:** done  

### Context & Scope
This is the core verification brain. Executing strictly on CPU threads to preserve GPU VRAM for Gemma, it accepts batches of `(Premise, Claim)` pairs, runs Softmax over cross-encoder logits, and evaluates against initial calibrated thresholds ($\tau_e \ge 0.70, \tau_c \ge 0.70$). It calculates document-level Hallucination Rate, Verification Rate, and Unverified Rate, and trips `is_safe = false` if any claim has $P(\text{contradiction}) \ge 0.85$.

### Acceptance Criteria
- [x] Model weights for `cross-encoder/nli-deberta-v3-base` load cleanly onto host CPU threads.
- [x] Input batches of `(Premise, Claim)` pairs return Softmax probabilities: $P(\text{entailment}) + P(\text{neutral}) + P(\text{contradiction}) = 1.0$.
- [x] Claims are classified into 3 discrete states based on configurable thresholds:
  - `SUPPORTED`: $P(\text{entailment}) \ge 0.70$
  - `CONTRADICTED`: $P(\text{contradiction}) \ge 0.70$
  - `UNVERIFIED`: neither threshold met, or premise is empty.
- [x] Document reliability metrics are computed accurately:
  - $\text{Hallucination Rate} = N_{\text{contradicted}} / N_{\text{total}} \times 100\%$
  - $\text{Verification Rate} = N_{\text{supported}} / N_{\text{total}} \times 100\%$
  - $\text{Unverified Rate} = N_{\text{unverified}} / N_{\text{total}} \times 100\%$
- [x] Safety Gate sets `is_safe = false` when any claim has $P(\text{contradiction}) \ge 0.85$.
- [x] Inference latency executes in $\le 80\text{ ms}$ per claim pair on modern multi-core CPU (`NFR-PERF-02`).
- [x] Unit tests (`tests/test_nli_verifier.py`) validate threshold edge cases, math calculations, and state assignments (`UT_VER_04`, `UT_VER_05`, `UT_VER_06`).

### Definition of Done
Engine exposes `NLIVerifier.verify_batch(pairs) -> VerificationReport`, passes all unit tests, and verifies 10 benchmark premise-claim pairs with expected labels.
