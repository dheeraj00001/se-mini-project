# 11 — Synthetic SSE Replay Streaming Engine

**What to build:**
A streaming response generator that activates when clients request `stream: true` on `POST /v1/chat/completions`, taking the finalized, verified, and mitigated response and emitting it as synthetic Server-Sent Event (SSE) chunks formatted to match the OpenAI streaming protocol.

**Sprint:** Sprint 2  
**Owner:** Student 1 (Gateway)  
**Story Points:** 3 SP  
**SRS Reference:** REQ-GW-03, REQ-GW-04, ADR-0008  
**Blocked by:** 10 — OpenAI-Compatible Reverse Proxy & Full Request Buffering  
**Status:** ready-for-agent  

### Context & Scope
Streaming clients (like chat web UIs or CLI streaming clients) fail if a proxy returns static JSON when `stream: true` is requested. To satisfy `ADR-0008` (full response verification without leaking hallucinated tokens), this engine splits the finalized, sanitized text into word/token chunks and replays them over an SSE `EventSourceResponse` with realistic pacing (~10–20 ms per token), terminating with `data: [DONE]`.

### Acceptance Criteria
- [ ] Requests to `POST /v1/chat/completions` with `stream: true` return `Content-Type: text/event-stream`.
- [ ] Response emits valid OpenAI streaming delta objects:
  `data: {"id": "...", "choices": [{"delta": {"content": "..."}}]}\n\n`
- [ ] Stream concludes with the standard terminal packet: `data: [DONE]\n\n`.
- [ ] No unverified or raw tokens ever leak to the client prior to full-document verification completion.
- [ ] Compatible with OpenAI Python SDK streaming iterator (`for chunk in response: ...`).
- [ ] Integration tests (`tests/test_sse_replay.py`) validate SSE framing, chunk integrity, and clean client disconnection (`UT_GW_04`, `ST_STR_01`, `ST_STR_02`).

### Definition of Done
Streaming requests via `curl -N` or OpenAI Python SDK yield progressive token display ending cleanly with `[DONE]`, verified via automated test suite.
