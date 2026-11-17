# Student 3 Contributions & Engineering Deliverables (UE24CS341A)

**Institution:** PES University, Bangalore  
**Department:** Department of Artificial Intelligence and Machine Learning  
**Course:** Software Engineering (UE24CS341A) — Mini-Project 2026  
**System:** Dehallucination Add-On Module Gateway  
**Student 3:** Harshad Patil ([@HarshadPatil108](https://github.com/HarshadPatil108))  
**Email:** harshadbussiness1234@gmail.com  
**Role:** Knowledge Systems & Retrieval Lead  
**Primary Subsystem:** Module 3: Hybrid Knowledge Ingestion & Retrieval Engine (`src/retrieval/`)  

---

## 1. Executive Summary

As the **Knowledge Systems & Retrieval Lead (Student 3)**, I designed and developed the complete knowledge ingestion and hybrid retrieval subsystem (Module 3) for the Dehallucination Add-On Module Gateway. Module 3 provides the foundational evidence grounding layer that powers factual verification. It parses and chunks heterogeneous reference documents (PDF, Markdown, TXT) into deterministic sliding windows, coordinates dual-index persistence across local file-backed Qdrant vector databases and disk-serialized Rank-BM25 inverted indices, merges multi-modal candidate rankings using Reciprocal Rank Fusion ($k=60$), and provides strictly opt-in Tier 3 external web grounding via DuckDuckGo with rigorous security air-gap enforcement.

All deliverables were developed and delivered following strict Agile test-driven development (TDD), tracked by dedicated GitHub issues, feature branches, pull requests, and automated test suites achieving $>85\%$ branch test coverage.

---

## 2. GitHub Traceability Matrix

| Backlog Ticket | GitHub Issue | Pull Request | Status | Key Deliverables & Code Seams |
| :--- | :---: | :---: | :---: | :--- |
| **Ticket 03**: Hybrid Knowledge Ingestion | [#13](https://github.com/dheeraj00001/se-mini-project/issues/13) | [#16](https://github.com/dheeraj00001/se-mini-project/pull/16) | **Merged** | • `src/retrieval/ingest.py`<br>• `src/retrieval/chunker.py`<br>• `src/retrieval/parser.py`<br>• `src/retrieval/store.py`<br>• `src/schemas/ingestion.py`<br>• `tests/test_ingestion.py`<br>• Multi-format extraction, SHA-256 chunk hashing, DualIndexStore persistence |
| **Ticket 04**: Hybrid Retrieval & RRF | [#14](https://github.com/dheeraj00001/se-mini-project/issues/14) | [#17](https://github.com/dheeraj00001/se-mini-project/pull/17) | **Merged** | • `src/retrieval/fusion.py`<br>• `src/retrieval/adapters.py`<br>• `src/retrieval/__init__.py`<br>• `tests/test_retrieval_adapters.py`<br>• RRF algorithm ($k=60$), Tier 1 context extraction, Tier 2 hybrid retrieval, sub-50ms latency |
| **Ticket 09**: Tier 3 Web Retrieval | [#15](https://github.com/dheeraj00001/se-mini-project/issues/15) | [#18](https://github.com/dheeraj00001/se-mini-project/pull/18) | **Merged** | • `src/retrieval/adapters.py`<br>• `src/gateway/middleware.py`<br>• `tests/test_web_retrieval.py`<br>• `docs/STUDENT_3_CONTRIBUTIONS.md`<br>• DuckDuckGo search integration, security ceiling (`SERVER_MAX_TIER`), air-gap safety |

---

## 3. Subsystem Architecture & Key Deliverables

### 3.1. Multi-Format Ingestion Pipeline & Deterministic Chunking (`src/retrieval/`)
- **SRS Requirements:** `REQ-RET-01`, `REQ-RET-02`, `REQ-RET-03`
- **Architecture Decision:** `ADR-0007` (Qdrant Dense + BM25 Sparse Hybrid Store)
- **Multi-Format Extraction:** Engineered extraction routines supporting PDF (`pypdf`), CommonMark/Markdown, and plain UTF-8 text files with defensive exception handling and encoding fallbacks.
- **Sliding-Window Chunking:** Implemented deterministic token-based chunking enforcing 250–300 token windows with a 50-token sliding overlap, ensuring semantic context continuity across chunk seams.
- **Deterministic SHA-256 Content Hashing:**
  $$\text{Chunk ID} = \text{SHA-256}(\text{source\_doc} \parallel \text{chunk\_index} \parallel \text{chunk\_text})$$
  Ensures identical content produces identical identifiers across repeated ingestion passes and guarantees synchronization between Qdrant vector points and BM25 index documents.
- **Duplicate Detection & Telemetry:** Built duplicate document skip detection (`has_document`) avoiding redundant embeddings computation, and added structured telemetry reporting via `get_ingestion_telemetry`.

### 3.2. DualIndexStore & Reciprocal Rank Fusion (`src/retrieval/store.py`, `src/retrieval/fusion.py`)
- **SRS Requirements:** `REQ-RET-03`, `REQ-RET-04`, `REQ-RET-05`, `NFR-PERF-01`
- **Architecture Decision:** `ADR-0002` (CPU Embeddings), `ADR-0004` (Waterfall Retrieval), `ADR-0007` (Hybrid Store)
- **Synchronized Persistence:** Coordinated local disk-backed Qdrant collections (`./data/qdrant_storage`) storing dense vector points with cosine distance distance metrics, and disk-serialized Rank-BM25 indices (`./data/bm25_index.joblib`).
- **Reciprocal Rank Fusion (RRF):**
  $$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
  where $k = 60$ and $r_m(d)$ is the 1-based rank of candidate chunk $d$ in retrieval modality $m$ (dense vector vs sparse keyword).
- **Deterministic Tie-Breaking:** Formatted fused candidates ordered descending by RRF score, breaking score ties lexicographically by deterministic `chunk_id`.
- **Latency Benchmark:** Achieved hybrid retrieval latency $\le 50\text{ ms}$ for local index searches (`NFR-PERF-01`).

### 3.3. Opt-In Tier 3 External Web Grounding (`src/retrieval/adapters.py`)
- **SRS Requirements:** `REQ-RET-05`, `REQ-RET-06`, `NFR-SEC-01`
- **Architecture Decision:** `ADR-0004` (Tier 3 Escalation and Privacy Preservation)
- **External Search Adapter:** Engineered asynchronous DuckDuckGo web search adapter (`Tier3WebAdapter`) querying live web premises when local grounding is inconclusive.
- **Security Tier Ceiling Enforcement:**
  $$\text{Execution Condition} \iff (\text{client\_max\_tier} == 3) \land (\text{SERVER\_MAX\_TIER} \ge 3)$$
  Guaranteed 0% external egress and complete dormancy when `client_max_tier <= 2` or `SERVER_MAX_TIER < 3`, satisfying enterprise air-gap compliance (`NFR-SEC-01`).
- **Graceful Error Handling:** Handled upstream search timeouts, rate limits, and network anomalies by safely returning empty candidates, causing downstream orchestrators to label the claim `UNVERIFIED` without crashing.

---

## 4. Test Verification & Quality Gates

| Test Suite File | Test Count | Focus Areas |
| :--- | :---: | :--- |
| `tests/test_ingestion.py` | 30 | PDF/MD/TXT parsing, sliding window chunk boundaries, deterministic SHA-256 hashing, DualIndexStore restart persistence, multipart/JSON API ingestion |
| `tests/test_retrieval_adapters.py` | 12 | Reciprocal Rank Fusion ($k=60$) mathematical properties, rank inversion resistance, Tier 1 context extraction, Tier 2 hybrid retrieval, sub-50ms latency |
| `tests/test_web_retrieval.py` | 26 | Tier 3 DuckDuckGo search integration, snippet parsing, security ceiling enforcement (`SERVER_MAX_TIER`), air-gap protection, async execution |

All 68 automated unit and integration tests across Module 3 pass with 100% green status and $>85\%$ branch coverage.
