# 04 — Hybrid Retrieval Adapters & Reciprocal Rank Fusion (Tiers 1 & 2)

**What to build:**
Retrieval adapters for in-context prompt passages (Tier 1) and local hybrid knowledge bases (Tier 2), querying Qdrant (dense vectors) and BM25 (sparse keywords) concurrently and merging their candidate rankings via Reciprocal Rank Fusion ($k=60$) to return the top candidate evidence premises.

**Sprint:** Sprint 1  
**Owner:** Student 3 (Retrieval)  
**Story Points:** 5 SP  
**SRS Reference:** REQ-RET-04, REQ-RET-05, ADR-0004, ADR-0007  
**Blocked by:** 03 — Hybrid Knowledge Ingestion & Deterministic Dual-Index Store  
**Status:** completed  

### Context & Scope
Given an atomic claim query, this module provides the retrieval adapter interface `RetrievalAdapter.search(claim: str, tier: int, context_docs: List[str]) -> List[EvidenceCandidate]`. It implements the mathematical Reciprocal Rank Fusion (RRF) algorithm to combine semantic matches from Qdrant with exact keyword hits from BM25 without requiring NLI logic (which is cleanly decoupled in Ticket 05 and orchestrated in Ticket 06).

### Acceptance Criteria
- [x] Tier 1 adapter extracts relevant passages from in-context documents provided in the request payload using lexical/embedding similarity.
- [x] Tier 2 adapter queries Qdrant for top-$N$ vector matches and BM25 for top-$N$ keyword matches given an atomic claim query.
- [x] RRF algorithm combines dense and sparse ranks with constant $k=60$:
  $$RRF(d) = \frac{1}{60 + \text{rank}_{\text{dense}}(d)} + \frac{1}{60 + \text{rank}_{\text{sparse}}(d)}$$
- [x] System returns top-$K$ reranked `EvidenceCandidate` objects containing chunk text, source metadata, and combined RRF score.
- [x] Retrieval latency completes in $\le 50\text{ ms}$ for local index queries.
- [x] Unit tests (`tests/test_retrieval_adapters.py`) verify RRF ranking math and adapter query execution (`UT_RET_04`).

### Definition of Done
Adapter module callable independently via unit tests, returning ranked evidence chunks from sample ingested corpus with verified RRF score ordering.
