# Team Contributions & Module Ownership (UE24CS341A)

**Institution:** PES University, Bangalore  
**Department:** Department of Artificial Intelligence and Machine Learning  
**Course:** Software Engineering (UE24CS341A) — Mini-Project 2026  
**System:** Dehallucination Add-On Module Gateway  
**Repository:** [https://github.com/dheeraj00001/se-mini-project](https://github.com/dheeraj00001/se-mini-project)  

---

## 1. Executive Summary & Team Structure

The **Dehallucination Add-On Module Gateway** is designed under strict modular separation of concerns. The team divided functional responsibilities into four core subsystems with typed interface boundaries, dedicated unit/integration test suites, independent agile backlog tickets, and Architecture Decision Records (ADRs).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DEHALLUCINATION ADD-ON GATEWAY                        │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│ Student             │ Engineering Role    │ Primary Subsystem               │
├─────────────────────┼─────────────────────┼─────────────────────────────────┤
│ **Student 1**       │ API Gateway &       │ **Module 1**: Gateway &         │
│ (Lead / Owner)      │ Streaming Architect │ Streaming Proxy Engine          │
├─────────────────────┼─────────────────────┼─────────────────────────────────┤
│ **Student 2**       │ NLP & Verification  │ **Module 2**: Claim             │
│                     │ Engine Lead         │ Segmentation & NLI Verification │
├─────────────────────┼─────────────────────┼─────────────────────────────────┤
│ **Student 3**       │ Knowledge Systems   │ **Module 3**: Hybrid Knowledge  │
│                     │ & Retrieval Lead    │ Ingestion & Retrieval Engine    │
├─────────────────────┼─────────────────────┼─────────────────────────────────┤
│ **Student 4**       │ Mitigation &        │ **Module 4**: Mitigation &      │
│                     │ Evaluation Lead     │ Two-Tier Benchmark Harness      │
└─────────────────────┴─────────────────────┴─────────────────────────────────┘
```

---

## 2. Detailed Student Contributions

### 👤 Student 1: API Gateway & Streaming Architect
* **Primary Owned Module**: `Module 1: src/gateway/`
* **Owned Requirements**: `REQ-GW-01`, `REQ-GW-02`, `REQ-GW-03`, `REQ-GW-04`, `REQ-GW-05`, `NFR-REL-01`, `NFR-PERF-02`
* **Governing ADRs**: `ADR-0001` (Dual-Mode Architecture), `ADR-0008` (Zero Token Leakage Full Buffering)
* **Assigned Backlog Tickets**: Ticket 01, Ticket 10, Ticket 11, Ticket 13 (Option B)

#### Core Engineering Deliverables
1. **FastAPI Application Factory & Middleware (`src/gateway/app.py`, `src/gateway/middleware.py`)**:
   - Built production-grade application factory with structured logging, process timing headers (`X-Process-Time-Ms`), and global exception handlers for validation (422) and server errors (500).
   - Configured open CORS policies allowing local web clients to interface with gateway endpoints without proxy hurdles.
   - Built dynamic security header override parser supporting `X-Dehallucination-Mitigation`, `X-Dehallucination-Max-Tier`, and legacy aliases.
2. **OpenAI-Compatible Reverse Proxy (`src/gateway/router.py`, `src/gateway/buffer.py`)**:
   - Engineered drop-in replacement endpoint at `POST /v1/chat/completions` accepting standard OpenAI schemas.
   - Enforced 100% full response buffering (`buffer_generator_response`) prior to releasing output, guaranteeing **0.00% token leakage** of unverified or contradicted propositions.
3. **Synthetic SSE Replay Streaming Engine (`src/gateway/sse.py`)**:
   - Implemented compliant Server-Sent Events (SSE) generator streaming sanitized text as progressive delta packets ending with `data: [DONE]`.
   - Engineered synthetic inter-token pacing at 15ms intervals (`X-SSE-Pacing-Ms`), delivering realistic interactive typewriter feedback without premature exposure to hallucinations.
4. **Standalone Verification Sidecar (`POST /verify`)**:
   - Implemented programmatic sidecar endpoint coordinating deterministic claim segmentation, tiered waterfall retrieval, and cross-encoder reliability scoring.
5. **Decoupled Awwwards-Tier React Frontend (`frontend/`)**:
   - Engineered modern React 19 + TypeScript + Vite + Tailwind CSS interface following `awwwards-hero-section` (Architecture B: Asymmetric Split) and `awwwards-sections` pipelines.
   - Interactive Live Playground for real-time verification and streaming chat completions.
   - Offline simulation fallback enabling immediate grading and demonstration without neural model downloads.

#### Code Seams Owned
- `src/gateway/app.py`
- `src/gateway/buffer.py`
- `src/gateway/middleware.py`
- `src/gateway/router.py`
- `src/gateway/sse.py`
- `src/main.py`
- `src/schemas/openai.py`
- `frontend/` (Full decoupled React application)

#### Automated Test Suites Owned
- `tests/test_gateway_scaffold.py` (Scaffolding, health check, schema validation)
- `tests/test_reverse_proxy.py` (OpenAI proxy buffering, error handling)
- `tests/test_proxy_router.py` (Header overrides, chat completions)
- `tests/test_sse_replay.py` (SSE format compliance, 15ms pacing, delta chunks)

---

### 👤 Student 2: NLP & Verification Engine Lead
* **Primary Owned Module**: `Module 2: src/verifier/`
* **Owned Requirements**: `REQ-VER-01`, `REQ-VER-02`, `REQ-VER-03`, `REQ-VER-04`, `REQ-VER-05`, `REQ-VER-06`, `NFR-PERF-01`
* **Governing ADRs**: `ADR-0004` (Deterministic Proposition Extraction), `ADR-0005` (CPU DeBERTa-v3 NLI), `ADR-0009` (Threshold Calibration)
* **Assigned Backlog Tickets**: Ticket 02, Ticket 05, Ticket 06

#### Core Engineering Deliverables
1. **Deterministic Proposition Extraction (`src/verifier/segmenter.py`)**:
   - Implemented rule-based sentence tokenization and syntactic clause splitting using NLTK `punkt` without non-deterministic LLM extractors.
   - Epistemic hedge detection (e.g. *"I think"*, *"arguably"*) ensuring underlying factual claims remain verifiable.
2. **CPU DeBERTa-v3 NLI Cross-Encoder (`src/verifier/nli.py`)**:
   - CPU-optimized `cross-encoder/nli-deberta-v3-base` inference with label mapping: Contradiction, Entailment, Neutral.
   - Calibrated decision thresholds ($\tau_e = 0.50$, $\tau_c = 0.50$) yielding 100% Contradiction Precision on held-out test splits.
   - Severe Contradiction Circuit Breaker ($\tau_{severe} \ge 0.85$ trips `is_safe = False`).
3. **Waterfall Orchestrator (`src/verifier/orchestrator.py`)**:
   - Document-level reliability calculation: Hallucination Rate, Verification Rate, and Unverified Rate.

#### Code Seams Owned
- `src/verifier/segmenter.py`
- `src/verifier/nli.py`
- `src/verifier/orchestrator.py`
- `src/schemas/verification.py`

#### Automated Test Suites Owned
- `tests/test_claim_segmenter.py` (Sentence/clause segmentation, epistemic hedges, latency benchmark)
- `tests/test_nli_verifier.py` (DeBERTa-v3 inference, probability normalization, safety gates)
- `tests/test_waterfall_e2e.py` (End-to-end verification orchestration)

---

### 👤 Student 3: Knowledge Systems & Retrieval Lead
* **Primary Owned Module**: `Module 3: src/retrieval/`
* **Owned Requirements**: `REQ-RET-01`, `REQ-RET-02`, `REQ-RET-03`, `REQ-RET-04`, `REQ-RET-05`, `REQ-RET-06`
* **Governing ADRs**: `ADR-0007` (Qdrant Dense + BM25 Sparse Hybrid Store)
* **Assigned Backlog Tickets**: Ticket 03, Ticket 04, Ticket 09

#### Core Engineering Deliverables
1. **Multi-Format Ingestion Pipeline (`src/retrieval/ingest.py`, `src/retrieval/chunker.py`, `src/retrieval/parser.py`)**:
   - Text extraction across PDF, Markdown, and TXT files with sliding window chunking (250–300 tokens, 50-token overlap).
   - Deterministic SHA-256 chunk identifier synchronization.
2. **DualIndexStore Hybrid Retrieval (`src/retrieval/store.py`, `src/retrieval/embedder.py`, `src/retrieval/fusion.py`)**:
   - Local disk-backed Qdrant collection with 384-dimensional cosine distance dense vectors.
   - Rank-BM25 inverted keyword index.
   - Reciprocal Rank Fusion ($k=60$) delivering 100% Recall@10 on gold evaluation splits.
3. **Tier 3 External Web Grounding (`src/retrieval/adapters.py`)**:
   - DuckDuckGo search integration with automated snippet extraction.
   - Server-side tier ceiling enforcement (`SERVER_MAX_TIER=2` default).

#### Code Seams Owned
- `src/retrieval/adapters.py`
- `src/retrieval/chunker.py`
- `src/retrieval/embedder.py`
- `src/retrieval/fusion.py`
- `src/retrieval/ingest.py`
- `src/retrieval/parser.py`
- `src/retrieval/store.py`
- `src/schemas/ingestion.py`

#### Automated Test Suites Owned
- `tests/test_ingestion.py` (Chunking, multi-format parsing, chunk hashing)
- `tests/test_retrieval_adapters.py` (Qdrant adapter, BM25 adapter, RRF fusion)
- `tests/test_web_retrieval.py` (DuckDuckGo retrieval, tier capping security)

---

### 👤 Student 4: Mitigation & Evaluation Lead
* **Primary Owned Module**: `Module 4: src/mitigation/` & `src/eval/`
* **Owned Requirements**: `REQ-MIT-01`, `REQ-MIT-02`, `REQ-MIT-03`, `REQ-MIT-04`, `REQ-MIT-05`, `REQ-MIT-06`, `NFR-QUAL-01`, `NFR-QUAL-02`, `NFR-QUAL-03`
* **Governing ADRs**: `ADR-0001` (4-Bit Local Gemma Allocation), `ADR-0006` (Targeted Rewriting), `ADR-0009` (Calibration Split Discipline)
* **Assigned Backlog Tickets**: Ticket 07, Ticket 08, Ticket 12

#### Core Engineering Deliverables
1. **Quantized Local Gemma Rewriter (`src/mitigation/generator.py`, `src/mitigation/rewriter.py`)**:
   - Local 4-bit Gemma 2B instruction model hosting targeting $\le 2.5\text{ GB}$ VRAM with CPU fallback.
   - Prompt-guided surgical clause correction using grounding premise evidence.
2. **Mitigation Pipeline & Fallback Cascade (`src/mitigation/pipeline.py`, `src/mitigation/fallback.py`)**:
   - Mandatory secondary NLI verification loop ensuring rewritten text does not introduce novel hallucinations.
   - Automated fallback to surgical span masking (`redact`) or hard refusal (`block`) if secondary verification fails.
3. **Two-Tier Benchmark & Evaluation Suite (`src/eval/`, `scripts/run_benchmark.py`, `tests/eval_local_50.py`)**:
   - Tier A 50-case synthetic edge-case regression battery executing in $\le 9\text{ s}$ on CPU.
   - Tier B empirical benchmark harness on FEVER and HaluEval slices with threshold grid search and scikit-learn metrics.

#### Code Seams Owned
- `src/mitigation/generator.py`
- `src/mitigation/rewriter.py`
- `src/mitigation/pipeline.py`
- `src/mitigation/fallback.py`
- `src/eval/benchmark.py`
- `src/eval/calibration.py`
- `src/eval/metrics.py`
- `scripts/run_benchmark.py`

#### Automated Test Suites Owned
- `tests/test_targeted_rewriter.py` (Gemma prompt builder, targeted rewriting)
- `tests/test_secondary_verification.py` (Secondary NLI loop, fallback cascade)
- `tests/test_eval_benchmark.py` (Calibration runner, metric calculations)
- `tests/eval_local_50.py` (50-case Tier A synthetic regression battery)

---

## 3. Contribution Matrix Summary

| Subsystem Component | Student 1 | Student 2 | Student 3 | Student 4 |
| :--- | :---: | :---: | :---: | :---: |
| **FastAPI Gateway & Routing** | **Lead** | Support | Support | - |
| **OpenAI Reverse Proxy & Buffering** | **Lead** | - | - | - |
| **Synthetic SSE Replay (15ms)** | **Lead** | - | - | - |
| **React Cybernetic Frontend (Option B)** | **Lead** | - | - | - |
| **Deterministic Proposition Extraction** | - | **Lead** | - | - |
| **CPU DeBERTa-v3 Cross-Encoder** | - | **Lead** | - | - |
| **Safety Gating & Circuit Breakers** | - | **Lead** | - | - |
| **Qdrant & BM25 Ingestion Store** | - | - | **Lead** | - |
| **Reciprocal Rank Fusion (RRF)** | - | - | **Lead** | - |
| **DuckDuckGo Web Search Fallback** | - | - | **Lead** | - |
| **Quantized Gemma 2B Rewriter** | - | - | - | **Lead** |
| **Secondary NLI Verification Loop** | - | - | - | **Lead** |
| **Tier A & B Benchmark Suite** | - | - | - | **Lead** |
