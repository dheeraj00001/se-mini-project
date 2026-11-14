# 02 — Deterministic Sentence & Clause Claim Segmenter

**What to build:**
A fast, deterministic NLP component that breaks raw AI-generated text into independent, atomic, verifiable claim strings using NLTK and syntactic heuristics without invoking any LLM passes, preserving epistemic hedges.

**Sprint:** Sprint 1  
**Owner:** Student 2 (Verification)  
**Story Points:** 3 SP  
**SRS Reference:** REQ-VER-01, REQ-VER-02, REQ-VER-03, NFR-PERF-01  
**Blocked by:** None — can start immediately  
**Status:** ready-for-agent  

### Context & Scope
NLI models require standalone hypothesis sentences. To keep latency under 5 ms and conserve GPU VRAM, text is tokenized into sentences via NLTK, followed by rule-based splitting of compound clauses having distinct subjects and verbs. Sentences with epistemic hedges (*"I think...", "According to sources..."*) must NOT be dropped; the underlying factual claim must be extracted for verification.

### Acceptance Criteria
- [ ] Segmenter splits multi-sentence paragraphs into individual sentences using `nltk.sent_tokenize`.
- [ ] Compound sentences with distinct independent clauses (e.g., *"The Eiffel Tower was built in 1889 and is located in Berlin"*) are split into distinct atomic claim strings.
- [ ] Epistemic hedges (*"I think"*, *"It is believed that"*) are stripped or retained such that the factual claim remains fully testable as an NLI hypothesis.
- [ ] Non-factual utterances (questions, standalone greetings, code blocks) are filtered out cleanly.
- [ ] Execution latency completes in $\le 5\text{ ms}$ for texts up to 500 tokens (`NFR-PERF-01`).
- [ ] Unit tests (`tests/test_claim_segmenter.py`) validate at least 15 edge cases (`UT_VER_01`, `UT_VER_02`, `UT_VER_03`).

### Definition of Done
Component callable as `ClaimSegmenter.extract_claims(text: str) -> List[AtomicClaim]`, fully covered by unit tests running green in `pytest`.
