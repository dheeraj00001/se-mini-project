# 03 — Hybrid Knowledge Ingestion & Deterministic Dual-Index Store

**What to build:**
A document ingestion service that extracts text from PDF, Markdown, and TXT files, chunks text into 250–300 token windows with 50-token overlap, calculates SHA-256 deterministic chunk IDs, and persists entries in both a local file-backed Qdrant vector store and a disk-serialized BM25 index.

**Sprint:** Sprint 1  
**Owner:** Student 3 (Retrieval)  
**Story Points:** 5 SP  
**SRS Reference:** REQ-RET-01, REQ-RET-02, REQ-RET-03, ADR-0007  
**Blocked by:** None — can start immediately  
**Status:** ready-for-agent  

### Context & Scope
For Tier 2 retrieval, reference knowledge must be searchable both semantically (dense embeddings) and lexically (exact keyword matches). Ingestion must ensure that both Qdrant and BM25 reference identical chunk boundaries and IDs, persisting data on disk to survive application restarts.

### Acceptance Criteria
- [ ] Ingestion pipeline accepts PDF, Markdown, and TXT files from a directory or via `POST /knowledge/ingest`.
- [ ] Documents are chunked into 250–300 token windows with a 50-token sliding overlap.
- [ ] Each chunk is assigned a deterministic SHA-256 hash ID derived from document metadata and chunk content.
- [ ] Dense embeddings are generated via the CPU embedding model (`embeddinggemma-2` / MiniLM) and indexed in local file-backed Qdrant (`./data/qdrant_storage`).
- [ ] Tokenized chunks are indexed in BM25 (`rank-bm25`) and serialized to disk (`./data/bm25_index.joblib`).
- [ ] Re-ingesting unchanged files detects duplicate hashes and skips re-computation.
- [ ] Unit tests (`tests/test_ingestion.py`) verify chunk boundaries, hash consistency, and index persistence across restarts (`UT_RET_01`, `UT_RET_02`, `UT_RET_03`).

### Definition of Done
A CLI script `python -m src.ingest --source ./docs` and API endpoint `POST /knowledge/ingest` successfully ingest sample PDFs/MDs, leaving persistent Qdrant and BM25 stores verified via `pytest`.
