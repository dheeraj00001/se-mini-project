"""Unit and system tests for Opt-In Tier 3 External Web Retrieval & Security Capping.

Covers RTM Traceability:
- UT_RET_05: Tier 3 Web Retrieval Adapter (DuckDuckGo), snippet parsing, URL citations,
  async execution, and graceful degradation on search errors (REQ-RET-05, ADR-0004).
- UT_RET_06: Server-side max tier ceiling enforcement (SERVER_MAX_TIER), air-gap protection,
  advisory logging, and complete dormancy when max_tier <= 2 (REQ-RET-06, NFR-SEC-01).
- ST_SEC_01: System-level security tier capping, X-Max-Tier & X-Mitigation-Mode header parsing,
  and graceful unverified degradation via Gateway POST /verify (REQ-GW-05, REQ-RET-06).
"""

import logging
import os
from typing import Any, List, Optional
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from duckduckgo_search.exceptions import (
    DuckDuckGoSearchException,
    RatelimitException,
    TimeoutException,
)
from src.gateway.app import create_app
from src.gateway import router as gateway_router
from src.gateway.middleware import (
    apply_header_overrides,
    enforce_server_tier_ceiling,
    get_server_max_tier,
    parse_security_headers,
)
from src.retrieval.adapters import (
    RetrievalAdapter,
    Tier3WebAdapter,
)
from src.retrieval.embedder import MockEmbedder
from src.retrieval.store import DualIndexStore
from src.schemas.verification import (
    ClaimState,
    EvidenceChunk,
    MitigationMode,
    VerificationRequest,
)
from src.verifier.orchestrator import WaterfallOrchestrator, get_orchestrator


class MockDDGSClient:
    """Mock DuckDuckGo search client simulating DDGS.text responses and errors."""

    def __init__(
        self,
        results: Optional[List[Any]] = None,
        exception_to_raise: Optional[Exception] = None,
    ) -> None:
        self.results: List[Any] = results if results is not None else []
        self.exception_to_raise = exception_to_raise
        self.call_count = 0
        self.last_keywords: Optional[str] = None
        self.last_max_results: Optional[int] = None

    def text(
        self,
        keywords: str,
        max_results: Optional[int] = None,
        region: Optional[str] = None,
        safesearch: str = "moderate",
        **kwargs: Any,
    ) -> List[Any]:
        self.call_count += 1
        self.last_keywords = keywords
        self.last_max_results = max_results
        if self.exception_to_raise is not None:
            raise self.exception_to_raise
        if max_results is not None:
            return self.results[:max_results]
        return self.results


# ============================================================================
# UT_RET_05: Tier 3 Web Retrieval Adapter & Snippet Parsing (REQ-RET-05)
# ============================================================================


def test_ut_ret_05_adapter_initialization_and_properties() -> None:
    """Verify Tier3WebAdapter default and custom properties."""
    mock_client = MockDDGSClient()
    adapter = Tier3WebAdapter(
        ddgs_client=mock_client,
        timeout=8.0,
        server_max_tier=3,
        region="uk-en",
        safesearch="off",
    )
    assert adapter.ddgs_client is mock_client
    assert adapter.timeout == 8.0
    assert adapter.server_max_tier == 3

    # Default init without arguments
    default_adapter = Tier3WebAdapter()
    assert default_adapter.timeout == 5.0
    assert default_adapter.server_max_tier in (1, 2, 3)


def test_ut_ret_05_search_successful_snippet_parsing() -> None:
    """Verify DuckDuckGo search results parse into structured EvidenceCandidate chunks."""
    raw_results = [
        {
            "title": "Paris - Official Capital of France",
            "href": "https://en.wikipedia.org/wiki/Paris",
            "body": "Paris is the capital and most populous city of France.",
        },
        {
            "title": "France Geography & Facts",
            "href": "https://www.britannica.com/place/France",
            "body": "The capital of France is Paris, located on the Seine River.",
        },
    ]
    mock_client = MockDDGSClient(results=raw_results)
    adapter = Tier3WebAdapter(ddgs_client=mock_client, server_max_tier=3)

    chunks = adapter.search("The capital of France is Paris.", limit=2)

    assert mock_client.call_count == 1
    assert mock_client.last_keywords == "The capital of France is Paris."
    assert mock_client.last_max_results == 2
    assert len(chunks) == 2

    c1 = chunks[0]
    assert isinstance(c1, EvidenceChunk)
    assert c1.tier == 3
    assert c1.source_doc == "https://en.wikipedia.org/wiki/Paris"
    assert "Paris is the capital and most populous city of France." in c1.text
    assert "Paris - Official Capital of France" in c1.text
    assert c1.score == 1.0  # rank 1 score: 1.0 / 1
    assert c1.metadata["url"] == "https://en.wikipedia.org/wiki/Paris"
    assert c1.metadata["title"] == "Paris - Official Capital of France"
    assert c1.metadata["source"] == "duckduckgo"
    assert c1.metadata["rank"] == 1
    assert len(c1.chunk_id) == 64  # deterministic SHA-256

    c2 = chunks[1]
    assert c2.tier == 3
    assert c2.source_doc == "https://www.britannica.com/place/France"
    assert c2.score == 0.5  # rank 2 score: 1.0 / 2
    assert c2.metadata["rank"] == 2


def test_ut_ret_05_search_empty_and_whitespace_claim() -> None:
    """Verify empty or whitespace query returns empty list without calling DuckDuckGo."""
    mock_client = MockDDGSClient()
    adapter = Tier3WebAdapter(ddgs_client=mock_client, server_max_tier=3)

    assert adapter.search("") == []
    assert adapter.search("   \n\t  ") == []
    assert mock_client.call_count == 0


def test_ut_ret_05_search_formatting_variations() -> None:
    """Verify robust parsing when title or body or href is missing/empty."""
    raw_results: List[Any] = [
        # Body only
        {"href": "https://example.com/1", "body": "Snippet text without title."},
        # Title only
        {"title": "Title Without Snippet Body", "href": "https://example.com/2"},
        # Missing href
        {"title": "No URL result", "body": "Content with fallback source doc."},
        # Empty body and title (must be skipped)
        {"href": "https://example.com/empty", "body": "", "title": ""},
        # Non-dict item (must be skipped)
        "invalid non-dict entry",
    ]
    mock_client = MockDDGSClient(results=raw_results)
    adapter = Tier3WebAdapter(ddgs_client=mock_client, server_max_tier=3)

    chunks = adapter.search("Variations query", limit=10)

    assert len(chunks) == 3
    assert chunks[0].text == "Snippet text without title."
    assert chunks[0].source_doc == "https://example.com/1"

    assert chunks[1].text == "Title Without Snippet Body"
    assert chunks[1].source_doc == "https://example.com/2"

    assert chunks[2].text == "No URL result: Content with fallback source doc."
    assert chunks[2].source_doc.startswith("duckduckgo_search_")


@pytest.mark.anyio
async def test_ut_ret_05_async_asearch_execution() -> None:
    """Verify async asearch() executes non-blockingly via asyncio.to_thread."""
    raw_results = [
        {
            "title": "Async Test Result",
            "href": "https://async.example.com",
            "body": "Async web retrieval works properly.",
        }
    ]
    mock_client = MockDDGSClient(results=raw_results)
    adapter = Tier3WebAdapter(ddgs_client=mock_client, server_max_tier=3)

    chunks = await adapter.asearch("Async claim", limit=1)

    assert len(chunks) == 1
    assert chunks[0].tier == 3
    assert chunks[0].source_doc == "https://async.example.com"
    assert mock_client.call_count == 1


def test_ut_ret_05_graceful_error_handling_ratelimit(caplog: pytest.LogCaptureFixture) -> None:
    """Verify RatelimitException and DuckDuckGoSearchException are caught gracefully."""
    mock_client = MockDDGSClient(exception_to_raise=RatelimitException("Rate limit hit"))
    adapter = Tier3WebAdapter(ddgs_client=mock_client, server_max_tier=3)

    with caplog.at_level(logging.WARNING):
        chunks = adapter.search("Rate limited query")

    assert chunks == []
    assert "DuckDuckGo search error" in caplog.text

    # Also verify base DuckDuckGoSearchException
    mock_client.exception_to_raise = DuckDuckGoSearchException("Base search failure")
    with caplog.at_level(logging.WARNING):
        chunks2 = adapter.search("Base failure query")
    assert chunks2 == []


def test_ut_ret_05_graceful_error_handling_timeout(caplog: pytest.LogCaptureFixture) -> None:
    """Verify TimeoutException is caught gracefully and returns empty list (REQ-RET-05)."""
    mock_client = MockDDGSClient(exception_to_raise=TimeoutException("Search timed out"))
    adapter = Tier3WebAdapter(ddgs_client=mock_client, server_max_tier=3)

    with caplog.at_level(logging.WARNING):
        chunks = adapter.search("Timeout query")

    assert chunks == []
    assert "DuckDuckGo search error" in caplog.text


def test_ut_ret_05_graceful_error_handling_generic_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify generic network ConnectionError or Exception is caught gracefully."""
    mock_client = MockDDGSClient(exception_to_raise=ConnectionError("Host unreachable"))
    adapter = Tier3WebAdapter(ddgs_client=mock_client, server_max_tier=3)

    with caplog.at_level(logging.WARNING):
        chunks = adapter.search("Network failure query")

    assert chunks == []
    assert "DuckDuckGo search error" in caplog.text


def test_ut_ret_05_retrieval_adapter_tier3_routing() -> None:
    """Verify unified RetrievalAdapter correctly dispatches tier=3 to Tier3WebAdapter."""
    raw_results = [
        {
            "title": "Unified Routing",
            "href": "https://unified.example.com",
            "body": "Routed to tier 3 adapter.",
        }
    ]
    mock_ddgs = MockDDGSClient(results=raw_results)
    tier3 = Tier3WebAdapter(ddgs_client=mock_ddgs, server_max_tier=3)
    adapter = RetrievalAdapter(tier3_adapter=tier3)

    assert adapter.tier3_adapter is tier3

    results = adapter.search("Routing test", tier=3, limit=1)
    assert len(results) == 1
    assert results[0].tier == 3
    assert results[0].source_doc == "https://unified.example.com"
    assert mock_ddgs.call_count == 1


@pytest.mark.anyio
async def test_ut_ret_05_retrieval_adapter_asearch_tier3_routing() -> None:
    """Verify unified RetrievalAdapter.asearch() delegates tier=3 asynchronously."""
    raw_results = [
        {
            "title": "Async Routing",
            "href": "https://async-routing.example.com",
            "body": "Async routing to tier 3 adapter.",
        }
    ]
    mock_ddgs = MockDDGSClient(results=raw_results)
    tier3 = Tier3WebAdapter(ddgs_client=mock_ddgs, server_max_tier=3)
    adapter = RetrievalAdapter(tier3_adapter=tier3)

    results = await adapter.asearch("Async routing test", tier=3, limit=1)
    assert len(results) == 1
    assert results[0].tier == 3
    assert mock_ddgs.call_count == 1

    # Also verify asearch on tier=1 or 2
    res_t2 = await adapter.asearch("T2 test", tier=2, limit=1)
    assert isinstance(res_t2, list)

    # Invalid tier
    with pytest.raises(ValueError):
        await adapter.asearch("Bad tier", tier=4)


# ============================================================================
# UT_RET_06: Server Max Tier Ceiling Enforcement & Dormancy (REQ-RET-06, NFR-SEC-01)
# ============================================================================


def test_ut_ret_06_server_max_tier_ceiling_blocks_tier3(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify Tier3WebAdapter blocks search when SERVER_MAX_TIER < 3 (REQ-RET-06, ADR-0004)."""
    mock_client = MockDDGSClient(
        results=[{"title": "Leak", "href": "https://leak.com", "body": "Leaked info"}]
    )
    # Server ceiling capped at 2 (e.g. enterprise air-gapped deployment)
    adapter = Tier3WebAdapter(ddgs_client=mock_client, server_max_tier=2)

    with caplog.at_level(logging.WARNING):
        chunks = adapter.search("Secret claim")

    assert chunks == []
    # Must never call external search
    assert mock_client.call_count == 0
    assert "Tier 3 search blocked: server max tier ceiling is 2" in caplog.text


def test_ut_ret_06_client_tier_below_3_blocks_search() -> None:
    """Verify Tier3WebAdapter blocks search when client explicitly requested tier != 3."""
    mock_client = MockDDGSClient(
        results=[{"title": "Leak", "href": "https://leak.com", "body": "Leaked info"}]
    )
    adapter = Tier3WebAdapter(ddgs_client=mock_client, server_max_tier=3)

    assert adapter.search("Secret claim", client_max_tier=2) == []
    assert adapter.search("Secret claim", client_max_tier=1) == []
    assert mock_client.call_count == 0


def test_ut_ret_06_orchestrator_caps_client_tier_with_advisory_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify orchestrator caps tier at server ceiling and logs advisory message (REQ-RET-06)."""
    mock_ddgs = MockDDGSClient(
        results=[
            {
                "title": "Moon Landing",
                "href": "https://nasa.gov",
                "body": "Apollo 11 landed on the Moon.",
            }
        ]
    )
    tier3_adapter = Tier3WebAdapter(ddgs_client=mock_ddgs, server_max_tier=2)
    retrieval_adapter = RetrievalAdapter(tier3_adapter=tier3_adapter)

    orchestrator = get_orchestrator(
        retrieval_adapter=retrieval_adapter,
        server_max_tier=2,
        use_mock=True,
    )

    req = VerificationRequest(
        text="Apollo 11 landed on the Moon.",
        max_retrieval_tier=3,  # Client asks for 3, but server ceiling is 2
    )

    with caplog.at_level(logging.WARNING):
        resp = orchestrator.verify(req)

    # Advisory log emitted
    assert (
        "Client requested retrieval tier 3 exceeding server ceiling 2; capping at 2" in caplog.text
    )
    # External search must remain completely dormant
    assert mock_ddgs.call_count == 0
    assert resp.claims[0].state == ClaimState.UNVERIFIED


def test_ut_ret_06_dormant_when_max_tier_1_or_2() -> None:
    """Verify Tier 3 adapter remains dormant when max_retrieval_tier <= 2 (NFR-SEC-01)."""
    mock_ddgs = MockDDGSClient(
        results=[{"title": "Test", "href": "https://test.com", "body": "Fact passage"}]
    )
    tier3_adapter = Tier3WebAdapter(ddgs_client=mock_ddgs, server_max_tier=3)
    retrieval_adapter = RetrievalAdapter(tier3_adapter=tier3_adapter)

    orchestrator = get_orchestrator(
        retrieval_adapter=retrieval_adapter,
        server_max_tier=3,
        use_mock=True,
    )

    # Case 1: Client specifies max_retrieval_tier=2
    req_t2 = VerificationRequest(
        text="Water boils at 100 degrees Celsius.",
        max_retrieval_tier=2,
    )
    orchestrator.verify(req_t2)
    assert mock_ddgs.call_count == 0

    # Case 2: Client specifies max_retrieval_tier=1
    req_t1 = VerificationRequest(
        text="Water boils at 100 degrees Celsius.",
        max_retrieval_tier=1,
    )
    orchestrator.verify(req_t1)
    assert mock_ddgs.call_count == 0


def test_ut_ret_06_env_var_server_max_tier_override() -> None:
    """Verify SERVER_MAX_TIER environment variable configures orchestrator ceiling."""
    with patch.dict(os.environ, {"SERVER_MAX_TIER": "1"}):
        orchestrator = WaterfallOrchestrator()
        assert orchestrator.server_max_tier == 1

    with patch.dict(os.environ, {"SERVER_MAX_TIER": "2"}):
        adapter = Tier3WebAdapter()
        assert adapter.server_max_tier == 2

    # Malformed env var falls back safely to 3
    with patch.dict(os.environ, {"SERVER_MAX_TIER": "not_an_int"}):
        fallback_orch = WaterfallOrchestrator()
        assert fallback_orch.server_max_tier == 3


def test_ut_ret_06_tier1_short_circuits_before_tier3() -> None:
    """Verify decisive Tier 1 evidence short-circuits and never queries Tier 3."""
    mock_ddgs = MockDDGSClient(
        results=[{"title": "Web Hit", "href": "https://web.com", "body": "Web snippet"}]
    )
    tier3_adapter = Tier3WebAdapter(ddgs_client=mock_ddgs, server_max_tier=3)
    retrieval_adapter = RetrievalAdapter(tier3_adapter=tier3_adapter)

    orchestrator = get_orchestrator(
        retrieval_adapter=retrieval_adapter,
        server_max_tier=3,
        use_mock=True,
    )

    # Decisive Tier 1 context provided
    req = VerificationRequest(
        text="The Eiffel Tower is in Paris.",
        context_documents=["The Eiffel Tower is located in Paris, France."],
        max_retrieval_tier=3,
    )
    resp = orchestrator.verify(req)

    assert resp.claims[0].state == ClaimState.SUPPORTED
    assert resp.claims[0].evidence_tier == 1
    # Tier 3 must NOT be called because Tier 1 short-circuited!
    assert mock_ddgs.call_count == 0


def test_ut_ret_06_tier2_short_circuits_before_tier3() -> None:
    """Verify decisive Tier 2 KB evidence short-circuits and never queries Tier 3."""
    store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))
    store.add_chunks(
        [
            EvidenceChunk(
                chunk_id="kb_mars",
                text="Mars is the fourth planet from the Sun.",
                source_doc="astronomy.txt",
                tier=2,
            )
        ]
    )

    mock_ddgs = MockDDGSClient(
        results=[{"title": "Web Hit", "href": "https://web.com", "body": "Web snippet"}]
    )
    tier3_adapter = Tier3WebAdapter(ddgs_client=mock_ddgs, server_max_tier=3)
    retrieval_adapter = RetrievalAdapter(store=store, tier3_adapter=tier3_adapter)

    orchestrator = get_orchestrator(
        store=store,
        retrieval_adapter=retrieval_adapter,
        server_max_tier=3,
        use_mock=True,
    )

    req = VerificationRequest(
        text="Mars is the fourth planet from the Sun.",
        context_documents=[],  # Tier 1 empty
        max_retrieval_tier=3,
    )
    resp = orchestrator.verify(req)

    assert resp.claims[0].state == ClaimState.SUPPORTED
    assert resp.claims[0].evidence_tier == 2
    # Tier 3 must NOT be called because Tier 2 was decisive!
    assert mock_ddgs.call_count == 0


def test_ut_ret_06_tier3_decisive_evidence_supports_claim() -> None:
    """Verify claim escalated to Tier 3 is verified as SUPPORTED with URL citation."""
    mock_ddgs = MockDDGSClient(
        results=[
            {
                "title": "Quantum Computing",
                "href": "https://nature.com/articles/quantum",
                "body": "Qubits exhibit quantum superposition and entanglement.",
            }
        ]
    )
    tier3_adapter = Tier3WebAdapter(ddgs_client=mock_ddgs, server_max_tier=3)
    retrieval_adapter = RetrievalAdapter(tier3_adapter=tier3_adapter)

    orchestrator = get_orchestrator(
        retrieval_adapter=retrieval_adapter,
        server_max_tier=3,
        use_mock=True,
    )

    req = VerificationRequest(
        text="Qubits exhibit quantum superposition and entanglement.",
        context_documents=[],
        max_retrieval_tier=3,
    )
    resp = orchestrator.verify(req)

    assert mock_ddgs.call_count == 1
    assert len(resp.claims) == 1
    claim = resp.claims[0]
    assert claim.state == ClaimState.SUPPORTED
    assert claim.evidence_tier == 3
    assert claim.evidence_text is not None
    assert "Qubits exhibit quantum superposition" in claim.evidence_text
    assert resp.is_safe is True
    assert resp.needs_review is False
    assert resp.verification_rate == 100.0


def test_ut_ret_06_tier3_network_failure_falls_back_to_unverified() -> None:
    """Verify search failure during Tier 3 escalation falls back to UNVERIFIED gracefully."""
    mock_ddgs = MockDDGSClient(exception_to_raise=TimeoutException("Connection timed out"))
    tier3_adapter = Tier3WebAdapter(ddgs_client=mock_ddgs, server_max_tier=3)
    retrieval_adapter = RetrievalAdapter(tier3_adapter=tier3_adapter)

    orchestrator = get_orchestrator(
        retrieval_adapter=retrieval_adapter,
        server_max_tier=3,
        use_mock=True,
    )

    req = VerificationRequest(
        text="Ancient civilizations utilized complex astronomical calendars.",
        max_retrieval_tier=3,
    )
    resp = orchestrator.verify(req)

    assert mock_ddgs.call_count == 1
    assert len(resp.claims) == 1
    claim = resp.claims[0]
    assert claim.state == ClaimState.UNVERIFIED
    assert claim.evidence_tier is None
    assert claim.evidence_text is None
    assert resp.is_safe is True
    assert resp.needs_review is True
    assert resp.unverified_rate == 100.0


# ============================================================================
# ST_SEC_01: System Security Tests & Header Capping via Gateway (REQ-GW-05, REQ-RET-06)
# ============================================================================


def test_st_sec_01_verify_endpoint_header_x_max_tier_capping(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ST_SEC_01: POST /verify with X-Max-Tier: 3 capped at SERVER_MAX_TIER=2 with advisory log."""
    mock_store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))
    mock_ddgs = MockDDGSClient(
        results=[{"title": "Web", "href": "https://web.com", "body": "Passage"}]
    )
    tier3 = Tier3WebAdapter(ddgs_client=mock_ddgs, server_max_tier=2)
    adapter = RetrievalAdapter(store=mock_store, tier3_adapter=tier3)
    orch = get_orchestrator(
        store=mock_store, retrieval_adapter=adapter, server_max_tier=2, use_mock=True
    )

    gateway_router._default_store = mock_store
    gateway_router._default_orchestrator = orch

    app = create_app()
    with TestClient(app) as client:
        with caplog.at_level(logging.WARNING):
            res = client.post(
                "/verify",
                json={"text": "Unknown historical proposition."},
                headers={"X-Max-Tier": "3"},
            )

    assert res.status_code == 200
    data = res.json()
    assert data["claims"][0]["state"] == "UNVERIFIED"
    # Web search was never invoked
    assert mock_ddgs.call_count == 0
    assert "capping at 2" in caplog.text


def test_st_sec_01_verify_endpoint_header_x_max_tier_opt_in_allowed() -> None:
    """ST_SEC_01: POST /verify with X-Max-Tier: 3 allowed when SERVER_MAX_TIER=3."""
    mock_store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))
    mock_ddgs = MockDDGSClient(
        results=[
            {
                "title": "Photosynthesis",
                "href": "https://biology.example.com",
                "body": "Plants convert sunlight into chemical energy.",
            }
        ]
    )
    tier3 = Tier3WebAdapter(ddgs_client=mock_ddgs, server_max_tier=3)
    adapter = RetrievalAdapter(store=mock_store, tier3_adapter=tier3)
    orch = get_orchestrator(
        store=mock_store, retrieval_adapter=adapter, server_max_tier=3, use_mock=True
    )

    gateway_router._default_store = mock_store
    gateway_router._default_orchestrator = orch

    app = create_app()
    with TestClient(app) as client:
        res = client.post(
            "/verify",
            # Body default max_retrieval_tier is 2, header overrides to 3!
            json={"text": "Plants convert sunlight into chemical energy."},
            headers={"X-Max-Tier": "3"},
        )

    assert res.status_code == 200
    data = res.json()
    assert mock_ddgs.call_count == 1
    assert data["claims"][0]["state"] == "SUPPORTED"
    assert data["claims"][0]["evidence_tier"] == 3


def test_st_sec_01_verify_endpoint_header_x_mitigation_mode_override() -> None:
    """ST_SEC_01: POST /verify with X-Mitigation-Mode header overriding body."""
    mock_store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))
    # Contradiction premise
    mock_store.add_chunks(
        [
            EvidenceChunk(
                chunk_id="kb_capital",
                text="Paris is the capital of France. Berlin is the capital of Germany.",
                source_doc="capitals.txt",
                tier=2,
            )
        ]
    )
    orch = get_orchestrator(store=mock_store, use_mock=True)
    gateway_router._default_store = mock_store
    gateway_router._default_orchestrator = orch

    app = create_app()
    with TestClient(app) as client:
        # Body specifies flag, but header overrides to redact!
        res = client.post(
            "/verify",
            json={
                "text": "The capital of France is Berlin.",
                "mitigation_mode": "flag",
            },
            headers={"X-Mitigation-Mode": "redact"},
        )

    assert res.status_code == 200
    data = res.json()
    assert data["claims"][0]["state"] == "CONTRADICTED"
    assert "[Claim redacted: Disproven by reference knowledge]" in data["sanitized_text"]


def test_st_sec_01_verify_endpoint_invalid_header_values() -> None:
    """ST_SEC_01: Verify 422 Unprocessable Entity on invalid X-Max-Tier or X-Mitigation-Mode."""
    app = create_app()
    with TestClient(app) as client:
        # Invalid tier > 3
        res_high = client.post(
            "/verify",
            json={"text": "Valid text."},
            headers={"X-Max-Tier": "4"},
        )
        assert res_high.status_code == 422
        assert "Invalid x-max-tier header" in res_high.json()["detail"]

        # Invalid non-integer tier
        res_nan = client.post(
            "/verify",
            json={"text": "Valid text."},
            headers={"X-Max-Tier": "tier_three"},
        )
        assert res_nan.status_code == 422

        # Invalid mitigation mode
        res_bad_mode = client.post(
            "/verify",
            json={"text": "Valid text."},
            headers={"X-Mitigation-Mode": "obliterate"},
        )
        assert res_bad_mode.status_code == 422
        assert "Invalid x-mitigation-mode header" in res_bad_mode.json()["detail"]


def test_st_sec_01_verify_endpoint_search_failure_graceful_200() -> None:
    """ST_SEC_01: POST /verify succeeds with 200 OK when external web search times out."""
    mock_store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))
    mock_ddgs = MockDDGSClient(exception_to_raise=TimeoutException("Gateway timeout to search"))
    tier3 = Tier3WebAdapter(ddgs_client=mock_ddgs, server_max_tier=3)
    adapter = RetrievalAdapter(store=mock_store, tier3_adapter=tier3)
    orch = get_orchestrator(
        store=mock_store, retrieval_adapter=adapter, server_max_tier=3, use_mock=True
    )

    gateway_router._default_store = mock_store
    gateway_router._default_orchestrator = orch

    app = create_app()
    with TestClient(app) as client:
        res = client.post(
            "/verify",
            json={"text": "Unverifiable claim requiring web lookup."},
            headers={"X-Max-Tier": "3"},
        )

    assert res.status_code == 200
    data = res.json()
    assert data["claims"][0]["state"] == "UNVERIFIED"
    assert data["is_safe"] is True
    assert data["needs_review"] is True


def test_middleware_helper_functions() -> None:
    """Direct tests for middleware functions: parse_security_headers, ceiling, etc."""
    # Empty headers
    assert parse_security_headers({}) == {}

    # Valid headers
    parsed = parse_security_headers({"X-Max-Tier": "2", "X-Mitigation-Mode": "REDACT"})
    assert parsed["max_retrieval_tier"] == 2
    assert parsed["mitigation_mode"] == MitigationMode.REDACT

    # apply_header_overrides with empty headers returns original
    req = VerificationRequest(text="Hello world")
    assert apply_header_overrides(req, {}) is req

    # apply_header_overrides with updates
    updated = apply_header_overrides(req, {"X-Max-Tier": "1", "X-Mitigation-Mode": "block"})
    assert updated.max_retrieval_tier == 1
    assert updated.mitigation_mode == MitigationMode.BLOCK

    # enforce_server_tier_ceiling
    eff, capped = enforce_server_tier_ceiling(client_tier=3, server_ceiling=2)
    assert eff == 2 and capped is True

    eff2, capped2 = enforce_server_tier_ceiling(client_tier=2, server_ceiling=3)
    assert eff2 == 2 and capped2 is False

    # get_server_max_tier
    with patch.dict(os.environ, {"SERVER_MAX_TIER": "2"}):
        assert get_server_max_tier() == 2
    with patch.dict(os.environ, {"SERVER_MAX_TIER": "invalid"}):
        assert get_server_max_tier() == 3
