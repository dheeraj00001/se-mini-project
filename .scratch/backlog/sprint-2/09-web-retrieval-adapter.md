# 09 — Opt-In Tier 3 External Web Retrieval (DuckDuckGo)

**What to build:**
An asynchronous external search adapter using `duckduckgo-search` that queries the live web for evidence snippets only when the client explicitly requests `max_retrieval_tier == 3` and the server-side ceiling (`SERVER_MAX_TIER >= 3`) permits it.

**Sprint:** Sprint 2  
**Owner:** Student 3 (Retrieval)  
**Story Points:** 3 SP  
**SRS Reference:** REQ-RET-05, REQ-RET-06, NFR-SEC-01, ADR-0004  
**Blocked by:** 04 — Hybrid Retrieval Adapters & Reciprocal Rank Fusion  
**Status:** completed  

### Context & Scope
When local knowledge (Tiers 1 & 2) fails to find relevant premises, the engine can escalate to Tier 3 web search. However, this is strictly opt-in to preserve user privacy and prevent data leakage in enterprise/air-gapped setups (`NFR-SEC-01`). If web search is disabled, unavailable, or yields irrelevant snippets, the claim is safely categorized as `UNVERIFIED`.

### Acceptance Criteria
- [x] Tier 3 adapter queries DuckDuckGo via async HTTP (`duckduckgo-search`) for top web snippets matching the claim.
- [x] Tier 3 search executes ONLY if `client_max_tier == 3` AND `SERVER_MAX_TIER >= 3`.
- [x] If `SERVER_MAX_TIER < 3`, client requests for Tier 3 are capped at the server ceiling with an advisory log.
- [x] Web search results are parsed into structured `EvidenceCandidate` chunks with URL citations.
- [x] Search errors (rate limits, network timeout) are caught gracefully and yield empty evidence, gracefully marking the claim `UNVERIFIED`.
- [x] Unit tests (`tests/test_web_retrieval.py`) mock HTTP calls and test security ceiling enforcement (`UT_RET_05`, `UT_RET_06`, `ST_SEC_01`).

### Definition of Done
Adapter returns cited web evidence snippets when opted in, and is proven to remain completely dormant when `max_tier <= 2`.
