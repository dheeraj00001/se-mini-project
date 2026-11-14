# 07 — Quantized Local Gemma GPU Generator & Targeted Rewriter

**What to build:**
A local generative pipeline that loads the 4-bit quantized Gemma model on GPU (`cuda:0`) within a 2.5 GB VRAM budget, capable of generating initial responses and performing prompt-guided targeted rewrites of `CONTRADICTED` claims ($P \ge 0.70$) using retrieved ground-truth premises without altering supported statements.

**Sprint:** Sprint 2  
**Owner:** Student 4 (Mitigation)  
**Story Points:** 5 SP  
**SRS Reference:** REQ-MIT-01, REQ-MIT-02, NFR-PERF-03, ADR-0002, ADR-0006  
**Blocked by:** 05 — CPU-Based DeBERTa-v3 NLI Verification  
**Status:** ready-for-agent  

### Context & Scope
When claims in a generated text contradict verified evidence, the rewriter must surgically fix only the contaminated statements. This module loads 4-bit quantized Gemma on GPU, applies a structured rewrite prompt passing the original text, contradicted claim, and grounding premise, and returns an updated coherent text. Unverified claims are explicitly preserved.

### Acceptance Criteria
- [ ] Gemma 4-bit weights load on `cuda:0` consuming $\le 2.5\text{ GB}$ VRAM (`NFR-PERF-03`).
- [ ] Rewriter constructs targeted correction prompts pairing each `CONTRADICTED` claim ($P \ge 0.70$) with its retrieved premise chunk.
- [ ] The generated rewrite replaces the contradicted assertion with the factual premise while retaining surrounding unmodified text.
- [ ] Claims classified as `UNVERIFIED` are never modified or erased by the rewrite prompt (`REQ-MIT-05`, `BR-01`).
- [ ] Rewriting pass executes in $\le 2.5\text{ s}$ on the RTX 2050 GPU.
- [ ] Unit tests (`tests/test_targeted_rewriter.py`) verify prompt construction, VRAM containment, and replacement accuracy (`UT_MIT_01`, `UT_MIT_02`).

### Definition of Done
Module exposes `GemmaRewriter.rewrite(original_text, contradicted_claims) -> str`, verified locally with test assertions checking that contradictory facts are replaced with premise facts.
