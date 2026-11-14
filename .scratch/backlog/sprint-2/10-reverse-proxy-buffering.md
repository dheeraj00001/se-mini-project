# 10 — OpenAI-Compatible Reverse Proxy & Full Request Buffering

**What to build:**
A drop-in reverse proxy endpoint at `POST /v1/chat/completions` that accepts standard OpenAI chat completion payloads, prompts the local Gemma model, buffers 100% of the raw response, runs full dehallucination verification and targeted mitigation, and returns a verified JSON response.

**Sprint:** Sprint 2  
**Owner:** Student 1 (Gateway)  
**Story Points:** 5 SP  
**SRS Reference:** REQ-GW-01, REQ-GW-03, REQ-GW-05, ADR-0001, ADR-0008  
**Blocked by:** 06 — NLI-Aware Waterfall Orchestration, 08 — Secondary Verification Loop  
**Status:** ready-for-agent  

### Context & Scope
This delivers the transparent proxy capability (Option C from ADR-0001). Existing applications using `openai.OpenAI(base_url="http://localhost:8000/v1")` can route chat completions through this endpoint without modifying their codebase. The proxy buffers the full response from Gemma to ensure complete sentence- and document-level verification before releasing output to the client.

### Acceptance Criteria
- [ ] Endpoint `POST /v1/chat/completions` accepts standard OpenAI schemas (`messages`, `model`, `temperature`, `max_tokens`).
- [ ] Prompts local Gemma on GPU to generate the raw response.
- [ ] Buffers the full raw response; extracts claims, runs waterfall retrieval, and executes NLI verification.
- [ ] Defaults to `mitigation_mode = rewrite`, correcting contradicted claims via Gemma and secondary verification before response packaging.
- [ ] Returns valid OpenAI `ChatCompletion` JSON structure (`choices[0].message.content`).
- [ ] Injects verification metadata headers (`X-Dehallucination-Safe`, `X-Hallucination-Rate`).
- [ ] Unit & integration tests (`tests/test_proxy_router.py`) verify compatibility using the official `openai` Python client SDK (`UT_GW_01`, `ST_E2E_01`).

### Definition of Done
Standard `curl` and Python `openai.chat.completions.create()` commands against `http://localhost:8000/v1` return verified and sanitized answers successfully.
