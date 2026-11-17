# Student 2 Contributions & Engineering Deliverables (UE24CS341A)

**Institution:** PES University, Bangalore  
**Department:** Department of Artificial Intelligence and Machine Learning  
**Course:** Software Engineering (UE24CS341A) — Mini-Project 2026  
**System:** Dehallucination Add-On Module Gateway  
**Student 2:** Balaji ([@avulabalaji35-rgb](https://github.com/avulabalaji35-rgb))  
**Email:** avulabalaji35@gmail.com  
**Role:** NLP & Verification Engine Lead  
**Primary Subsystem:** Module 2: Claim Segmentation & NLI Verification Engine (`src/verifier/`)  

---

## 1. Executive Summary

As the **NLP & Verification Engine Lead (Student 2)**, I was responsible for designing and implementing the core verification brain of the Dehallucination Add-On Module Gateway. Module 2 deterministically segments unstructured LLM outputs into atomic propositions, queries cascading evidence tiers, performs 3-way Natural Language Inference (NLI) classification on CPU, evaluates severe contradiction circuit breakers, and compiles document-level reliability telemetry.

All deliverables were developed under test-driven agile discipline, with dedicated GitHub issues, feature branches, pull requests, and automated test suites.

---

## 2. GitHub Traceability Matrix

| Backlog Ticket | GitHub Issue | Pull Request | Status | Key Deliverables & Code Seams |
| :--- | :---: | :---: | :---: | :--- |
| **Ticket 02**: Deterministic Claim Segmenter | [#7](https://github.com/dheeraj00001/se-mini-project/issues/7) | [#10](https://github.com/dheeraj00001/se-mini-project/pull/10) | **Merged** | • `src/verifier/segmenter.py`<br>• `tests/test_claim_segmenter.py`<br>• NLTK sentence splitting, syntactic clause decomposition, epistemic hedge preservation |
| **Ticket 05**: CPU DeBERTa-v3 NLI Engine | [#8](https://github.com/dheeraj00001/se-mini-project/issues/8) | [#11](https://github.com/dheeraj00001/se-mini-project/pull/11) | **Merged** | • `src/verifier/nli.py`<br>• `src/schemas/verification.py`<br>• `tests/test_nli_verifier.py`<br>• Cross-encoder inference, Softmax normalization, circuit breaker ($\tau_{severe} \ge 0.85$) |
| **Ticket 06**: Waterfall Orchestration MVP | [#9](https://github.com/dheeraj00001/se-mini-project/issues/9) | [#12](https://github.com/dheeraj00001/se-mini-project/pull/12) | **Merged** | • `src/verifier/orchestrator.py`<br>• `tests/test_waterfall_e2e.py`<br>• Sequential tiered retrieval loop, early short-circuiting, telemetry summary |

---

## 3. Subsystem Architecture & Key Deliverables

### 3.1. Deterministic Proposition Extraction (`src/verifier/segmenter.py`)
- **SRS Requirements:** `REQ-VER-01`, `REQ-VER-02`, `REQ-VER-03`, `NFR-PERF-01`
- **Architecture Decision:** `ADR-0004` (Deterministic Proposition Extraction)
- **Design Philosophy:** Avoided non-deterministic recursive LLM calls for claim extraction. Implemented rule-based sentence tokenization (`nltk.sent_tokenize`) followed by syntactic clause splitting.
- **Syntactic Splitting:** Handled compound sentences joined by coordinating conjunctions (`and`, `but`, `while`, `whereas`) and compound predicates, ensuring each resulting claim is a standalone verifiable assertion.
- **Epistemic Hedge Detection:** Engineered regex filters to strip epistemic belief prefixes (*"I think"*, *"It is believed that"*, *"Preliminary data suggests"*, *"Purportedly"*) so that underlying factual claims remain testable hypothesis strings.
- **Latency Benchmark:** Achieved latency $\le 5\text{ ms}$ for 500-token paragraphs (`NFR-PERF-01`).

### 3.2. CPU DeBERTa-v3 Cross-Encoder Verification (`src/verifier/nli.py`)
- **SRS Requirements:** `REQ-VER-04`, `REQ-VER-05`, `REQ-VER-06`, `NFR-PERF-02`
- **Architecture Decision:** `ADR-0002` (CPU Thread Execution), `ADR-0005` (Cross-Encoder Safety Gating)
- **Model Topology:** Executed `cross-encoder/nli-deberta-v3-base` strictly on host CPU threads to preserve GPU VRAM for the 4-bit Gemma generator.
- **Softmax Normalization:**
  $$\text{Softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$
  Guaranteed continuous probabilities with $\sum P = 1.0$ and label mapping `[0: contradiction, 1: entailment, 2: neutral]`.
- **Decision Thresholds:** Calibrated $\tau_e = 0.50$ and $\tau_c = 0.50$ on held-out splits, achieving 100% Contradiction Precision.
- **Severe Contradiction Circuit Breaker:**
  $$\text{if } P(\text{contradiction}) \ge 0.85 \implies \text{is\_safe} = \text{False}$$
  Immediate safety trip preventing dangerous hallucinations from passing through the gateway.

### 3.3. Waterfall Orchestration & Document Reliability (`src/verifier/orchestrator.py`)
- **SRS Requirements:** `REQ-GW-02`, `REQ-RET-05`
- **Architecture Decision:** `ADR-0004` (Cascading Waterfall Verification)
- **Sequential Tiering:**
  $$\text{Tier 1: In-Context} \xrightarrow{\text{Inconclusive}} \text{Tier 2: Local KB (Qdrant + BM25)} \xrightarrow{\text{Inconclusive}} \text{Tier 3: External Web Search}$$
- **Short-Circuit Optimization:** Decisive evidence ($P(\text{entailment}) \ge \tau_e$ or $P(\text{contradiction}) \ge \tau_c$) terminates the retrieval loop immediately.
- **Document-Level Reliability Metrics:**
  $$\text{Hallucination Rate} = \frac{N_{\text{contradicted}}}{N_{\text{total}}} \times 100\%$$
  $$\text{Verification Rate} = \frac{N_{\text{supported}}}{N_{\text{total}}} \times 100\%$$
  $$\text{Unverified Rate} = \frac{N_{\text{unverified}}}{N_{\text{total}}} \times 100\%$$

---

## 4. Test Verification & Quality Gates

| Test File | Test Count | Focus Areas |
| :--- | :---: | :--- |
| `tests/test_claim_segmenter.py` | 23 | Sentence tokenization, clause splitting, epistemic hedges, non-factual filtering, latency $\le 5$ ms |
| `tests/test_nli_verifier.py` | 39 | DeBERTa label mapping, Softmax bounds, decision thresholds, severe contradiction circuit breaker ($\tau \ge 0.85$) |
| `tests/test_waterfall_e2e.py` | 18 | Sequential cascading retrieval, early short-circuiting, telemetry compilation, schema compliance |
