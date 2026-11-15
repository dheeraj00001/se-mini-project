"""Unit and Integration Tests for Synthetic SSE Replay Streaming Engine (Ticket 11).

Covers:
- UT_GW_04: SSE chunk schemas, token splitting, SSE event framing, pacing (REQ-GW-04, ADR-0008).
- ST_STR_01: Full request buffering and zero unverified token leakage during streaming
  (ADR-0008, CON-04).
- ST_STR_02: Official OpenAI SDK streaming iterator compatibility (Sync & Async)
  and client disconnect handling.
"""

import json
import time
from typing import Generator, List
import httpx
import openai
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.gateway.app import create_app
from src.gateway.router import set_generator_instance, set_orchestrator_instance
from src.gateway.sse import (
    SyntheticSSEReplay,
    format_sse_event,
    set_sse_pacing_delay_ms,
    split_text_into_tokens,
)
from src.mitigation.fallback import DEFAULT_REDACTION_TOKEN
from src.mitigation.generator import MockGenerator
from src.retrieval.embedder import MockEmbedder
from src.retrieval.store import DualIndexStore
from src.schemas.openai import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
)
from src.schemas.verification import EvidenceChunk
from src.verifier.nli import MockNLIVerifier
from src.verifier.orchestrator import WaterfallOrchestrator, get_orchestrator


class SyncTestClientTransport(httpx.BaseTransport):
    """Bridge official sync OpenAI client to in-memory FastAPI TestClient."""

    def __init__(self, test_client: TestClient) -> None:
        self._test_client = test_client

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = request.url.path
        if request.url.query:
            url += "?" + request.url.query.decode("utf-8")
        res = self._test_client.request(
            method=request.method,
            url=url,
            content=request.content,
            headers=dict(request.headers),
        )
        return httpx.Response(
            status_code=res.status_code,
            headers=res.headers,
            content=res.content,
            request=request,
        )


@pytest.fixture
def app() -> FastAPI:
    """Create a fresh FastAPI app instance."""
    return create_app()


@pytest.fixture
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    """Provide a TestClient with clean default dependencies."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_gen() -> MockGenerator:
    """Provide a configured deterministic MockGenerator."""
    return MockGenerator(model_name="mock-gemma-4-E4B", simulated_vram_gb=2.2)


@pytest.fixture
def configured_orchestrator(mock_gen: MockGenerator) -> WaterfallOrchestrator:
    """Provide an orchestrator backed by in-memory DualIndexStore and MockNLIVerifier."""
    store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))
    store.add_chunks(
        [
            EvidenceChunk(
                chunk_id="chunk-paris-01",
                text="Paris is the official capital and largest city of France.",
                source_doc="geography_encyclopedia.txt",
                tier=2,
            ),
            EvidenceChunk(
                chunk_id="chunk-eiffel-01",
                text="The Eiffel Tower was designed by Gustave Eiffel for the 1889 Exposition.",
                source_doc="geography_handbook.txt",
                tier=2,
            ),
        ]
    )

    nli = MockNLIVerifier()
    orch = get_orchestrator(
        store=store,
        nli_verifier=nli,
        use_mock=True,
        enable_mitigation=True,
    )
    return orch


@pytest.fixture(autouse=True)
def reset_gateway_dependencies(
    configured_orchestrator: WaterfallOrchestrator, mock_gen: MockGenerator
) -> Generator[None, None, None]:
    """Automatically configure and teardown router dependencies per test."""
    set_orchestrator_instance(configured_orchestrator)
    set_generator_instance(mock_gen)
    yield
    set_orchestrator_instance(None)
    set_generator_instance(None)


# ============================================================================
# UT_GW_04: Schema, Token Splitting, Framing, and Pacing Unit Tests
# ============================================================================


def test_ut_gw_04_chunk_schemas_and_validation() -> None:
    """UT_GW_04: Validate ChatCompletionChunk, Choice, and Delta Pydantic schemas."""
    delta = ChatCompletionChunkDelta(role="assistant", content="Hello")
    assert delta.role == "assistant"
    assert delta.content == "Hello"

    choice = ChatCompletionChunkChoice(index=0, delta=delta, finish_reason=None)
    assert choice.index == 0
    assert choice.finish_reason is None

    chunk = ChatCompletionChunk(
        id="chatcmpl-test-123",
        model="google/gemma-4-E4B",
        choices=[choice],
    )
    assert chunk.id == "chatcmpl-test-123"
    assert chunk.object == "chat.completion.chunk"
    assert chunk.model == "google/gemma-4-E4B"
    assert chunk.created > 0
    assert len(chunk.choices) == 1

    # Verify JSON serialization compliance with OpenAI format
    dumped = json.loads(chunk.model_dump_json())
    assert dumped["id"] == "chatcmpl-test-123"
    assert dumped["object"] == "chat.completion.chunk"
    assert dumped["choices"][0]["delta"]["content"] == "Hello"


def test_ut_gw_04_split_text_into_tokens() -> None:
    """UT_GW_04: Verify splitting text preserves full text integrity across whitespaces."""
    # Standard sentence
    text = "The capital of France is Paris."
    tokens = split_text_into_tokens(text)
    assert len(tokens) > 1
    assert "".join(tokens) == text

    # Multiline text with punctuation
    multiline = "# Introduction\n\nDehallucination prevents fabricated claims.\n"
    tokens_multi = split_text_into_tokens(multiline)
    assert "".join(tokens_multi) == multiline

    # Leading and trailing whitespaces
    spaced = "   spaces at edges   "
    assert "".join(split_text_into_tokens(spaced)) == spaced

    # Empty string
    assert split_text_into_tokens("") == []


def test_ut_gw_04_format_sse_event() -> None:
    """UT_GW_04: Verify SSE formatting produces proper data: ...\\n\\n frames."""
    # Terminal DONE marker
    assert format_sse_event("[DONE]") == "data: [DONE]\n\n"

    # String payload
    assert format_sse_event("hello world") == "data: hello world\n\n"

    # Dict payload
    payload = {"status": "ok", "count": 1}
    formatted = format_sse_event(payload)
    assert formatted.startswith("data: ")
    assert formatted.endswith("\n\n")
    assert json.loads(formatted[6:-2]) == payload

    # Pydantic model payload
    chunk = ChatCompletionChunk(
        id="chatcmpl-sse-fmt",
        model="test-model",
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(content="test"),
                finish_reason=None,
            )
        ],
    )
    sse_line = format_sse_event(chunk)
    assert sse_line.startswith("data: {")
    assert sse_line.endswith("}\n\n")
    parsed = json.loads(sse_line[6:-2])
    assert parsed["id"] == "chatcmpl-sse-fmt"
    assert parsed["choices"][0]["delta"]["content"] == "test"


def test_ut_gw_04_synthetic_replay_chunk_sequence() -> None:
    """UT_GW_04: Verify SyntheticSSEReplay emits role chunk, token chunks, stop chunk,
    and [DONE]."""
    engine = SyntheticSSEReplay(pacing_delay_ms=0.0)
    text = "Paris is capital."
    events = list(engine.iter_sse_events(text=text, model="google/gemma-4-E4B"))

    # Sequence structure:
    # 0: initial role chunk
    # 1..N: content tokens
    # N+1: finish_reason='stop' chunk
    # N+2: data: [DONE]\n\n
    assert len(events) >= 4
    assert events[-1] == "data: [DONE]\n\n"

    # First event should establish assistant role
    first_data = json.loads(events[0][6:-2])
    assert first_data["choices"][0]["delta"]["role"] == "assistant"

    # Final chunk before DONE should have finish_reason='stop'
    stop_data = json.loads(events[-2][6:-2])
    assert stop_data["choices"][0]["finish_reason"] == "stop"

    # Reconstructed content from intermediate chunks
    reconstructed = ""
    for ev in events[:-1]:
        d = json.loads(ev[6:-2])
        c = d["choices"][0]["delta"].get("content")
        if c:
            reconstructed += c
    assert reconstructed == text


def test_ut_gw_04_synthetic_replay_empty_text() -> None:
    """UT_GW_04: Verify SyntheticSSEReplay handles empty text gracefully."""
    engine = SyntheticSSEReplay(pacing_delay_ms=0.0)
    events = list(engine.iter_sse_events(text="", model="google/gemma-4-E4B"))

    # Should contain role chunk, stop chunk, and [DONE]
    assert len(events) == 3
    assert events[-1] == "data: [DONE]\n\n"
    first_data = json.loads(events[0][6:-2])
    assert first_data["choices"][0]["delta"]["role"] == "assistant"
    stop_data = json.loads(events[1][6:-2])
    assert stop_data["choices"][0]["finish_reason"] == "stop"


@pytest.mark.anyio
async def test_ut_gw_04_pacing_delay_timing() -> None:
    """UT_GW_04: Verify pacing delay introduces realistic elapsed time between chunks."""
    # 5 words with 15ms pacing = ~75ms
    engine_paced = SyntheticSSEReplay(pacing_delay_ms=15.0)
    t0 = time.perf_counter()
    chunks_paced: List[str] = []
    async for ev in engine_paced.aiter_sse_events(
        text="One two three four five",
        model="google/gemma-4-E4B",
    ):
        chunks_paced.append(ev)
    elapsed_paced = time.perf_counter() - t0

    assert len(chunks_paced) >= 6
    # 5 tokens * 15ms = 75ms; assert at least 50ms elapsed
    assert elapsed_paced >= 0.05

    # 0ms pacing should be near instantaneous
    engine_instant = SyntheticSSEReplay(pacing_delay_ms=0.0)
    t1 = time.perf_counter()
    chunks_instant: List[str] = []
    async for ev in engine_instant.aiter_sse_events(
        text="One two three four five",
        model="google/gemma-4-E4B",
    ):
        chunks_instant.append(ev)
    elapsed_instant = time.perf_counter() - t1

    assert elapsed_instant < 0.05


# ============================================================================
# ST_STR_01: Zero-Leakage & Full Interception Streaming Tests
# ============================================================================


def test_st_str_01_zero_token_leakage_streaming_mitigation_rewrite(
    client: TestClient,
    mock_gen: MockGenerator,
    configured_orchestrator: WaterfallOrchestrator,
) -> None:
    """ST_STR_01: Verify contradictory text is intercepted and rewritten before SSE emission."""
    set_sse_pacing_delay_ms(0.0)

    # Raw model generates a contradicted statement
    mock_gen.register_response("capital of France", "Berlin is the official capital of France.")

    payload = {
        "model": "google/gemma-4-E4B",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True,
        "mitigation_mode": "rewrite",
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    # Verification headers must be present even for streaming responses
    assert "X-Dehallucination-Safe" in response.headers
    assert "X-Hallucination-Rate" in response.headers
    assert "X-Verification-Rate" in response.headers

    # CON-04: The contradicted token 'Berlin' must NEVER leak out in any chunk
    assert "Berlin" not in response.text
    # Reassemble deltas
    reconstructed = "".join(
        json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )
    # ADR-0008: The verified/rewritten token 'Paris' must be emitted
    assert "Paris" in reconstructed
    assert response.text.endswith("data: [DONE]\n\n")


def test_st_str_01_zero_token_leakage_streaming_mitigation_redact(
    client: TestClient,
    mock_gen: MockGenerator,
    configured_orchestrator: WaterfallOrchestrator,
) -> None:
    """ST_STR_01: Verify contradictory text is redacted into standard redaction token."""
    set_sse_pacing_delay_ms(0.0)

    mock_gen.register_response("capital of France", "Berlin is the official capital of France.")

    payload = {
        "model": "google/gemma-4-E4B",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True,
        "mitigation_mode": "redact",
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    # CON-04: The contradicted token 'Berlin' must NEVER leak out
    assert "Berlin" not in response.text
    reconstructed = "".join(
        json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )
    assert DEFAULT_REDACTION_TOKEN in reconstructed
    assert response.text.endswith("data: [DONE]\n\n")


def test_st_str_01_zero_token_leakage_streaming_mitigation_block(
    client: TestClient,
    mock_gen: MockGenerator,
    configured_orchestrator: WaterfallOrchestrator,
) -> None:
    """ST_STR_01: Verify block mode suppresses output and raises 422 before streaming."""
    set_sse_pacing_delay_ms(0.0)

    mock_gen.register_response("capital of France", "Berlin is the official capital of France.")

    payload = {
        "model": "google/gemma-4-E4B",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True,
        "mitigation_mode": "block",
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 422
    assert "factually grounded response" in response.json()["detail"].lower()


# ============================================================================
# ST_STR_02: Official OpenAI SDK (Sync & Async) & Disconnection Tests
# ============================================================================


def test_st_str_02_official_openai_sdk_sync_streaming(
    client: TestClient,
    mock_gen: MockGenerator,
    configured_orchestrator: WaterfallOrchestrator,
) -> None:
    """ST_STR_02: Validate stream consumption via official sync openai.OpenAI SDK."""
    set_sse_pacing_delay_ms(0.0)

    mock_gen.register_response(
        "capital of France",
        "Paris is the official capital of France.",
    )

    sync_http = httpx.Client(transport=SyncTestClientTransport(client))
    oai_client = openai.OpenAI(
        api_key="mock-api-key",
        base_url="http://test/v1",
        http_client=sync_http,  # type: ignore[arg-type]
    )

    stream = oai_client.chat.completions.create(
        model="google/gemma-4-E4B",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stream=True,
    )

    collected_content: List[str] = []
    finish_reasons: List[str] = []

    for chunk in stream:
        assert chunk.object == "chat.completion.chunk"
        assert chunk.model == "google/gemma-4-E4B"
        for choice in chunk.choices:
            if choice.delta.content:
                collected_content.append(choice.delta.content)
            if choice.finish_reason:
                finish_reasons.append(choice.finish_reason)

    assembled = "".join(collected_content)
    assert "Paris" in assembled
    assert "stop" in finish_reasons


@pytest.mark.anyio
async def test_st_str_02_official_openai_sdk_async_streaming(
    app: FastAPI,
    mock_gen: MockGenerator,
    configured_orchestrator: WaterfallOrchestrator,
) -> None:
    """ST_STR_02: Validate stream consumption via official async openai.AsyncOpenAI SDK."""
    set_sse_pacing_delay_ms(0.0)

    mock_gen.register_response(
        "capital of France",
        "Paris is the official capital of France.",
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as http_client:
        async_oai = openai.AsyncOpenAI(
            api_key="mock-api-key",
            base_url="http://test/v1",
            http_client=http_client,  # type: ignore[arg-type]
        )

        stream = await async_oai.chat.completions.create(
            model="google/gemma-4-E4B",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            stream=True,
        )

        collected_content: List[str] = []
        finish_reasons: List[str] = []

        async for chunk in stream:
            assert chunk.object == "chat.completion.chunk"
            assert chunk.model == "google/gemma-4-E4B"
            for choice in chunk.choices:
                if choice.delta.content:
                    collected_content.append(choice.delta.content)
                if choice.finish_reason:
                    finish_reasons.append(choice.finish_reason)

        assembled = "".join(collected_content)
        assert "Paris" in assembled
        assert "stop" in finish_reasons


@pytest.mark.anyio
async def test_st_str_02_client_disconnect_handling() -> None:
    """ST_STR_02: Verify server aborts SSE generation early if client disconnects."""
    engine = SyntheticSSEReplay(pacing_delay_ms=5.0)

    # Create a mock Request whose is_disconnected() toggles to True after 2 checks
    disconnect_counter = 0

    class MockDisconnectRequest:
        async def is_disconnected(self) -> bool:
            nonlocal disconnect_counter
            disconnect_counter += 1
            return disconnect_counter >= 3

    emitted_chunks: List[str] = []
    long_text = "One two three four five six seven eight nine ten eleven twelve thirteen."

    async for chunk in engine.aiter_sse_events(
        text=long_text,
        model="google/gemma-4-E4B",
        request=MockDisconnectRequest(),  # type: ignore[arg-type]
    ):
        emitted_chunks.append(chunk)

    # Should have stopped early and NOT reached data: [DONE]\n\n
    assert len(emitted_chunks) < 10
    assert emitted_chunks[-1] != "data: [DONE]\n\n"


def test_st_str_02_curl_n_framing_and_headers(
    client: TestClient,
    mock_gen: MockGenerator,
    configured_orchestrator: WaterfallOrchestrator,
) -> None:
    """ST_STR_02: Verify raw HTTP streaming response matches curl -N and SSE standard."""
    set_sse_pacing_delay_ms(0.0)

    mock_gen.register_response("capital of France", "Paris is the official capital of France.")

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "stream": True,
        },
        headers={"X-SSE-Pacing-Ms": "0"},
    )
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("text/event-stream")
    assert response.headers.get("Cache-Control") == "no-cache"
    assert response.headers.get("Connection") == "keep-alive"

    lines = response.text.split("\n\n")
    # Non-empty lines
    non_empty = [line for line in lines if line.strip()]
    assert len(non_empty) >= 3
    for line in non_empty:
        assert line.startswith("data: ")

    assert non_empty[-1] == "data: [DONE]"
