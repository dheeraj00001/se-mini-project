"""Unit and Integration Tests for OpenAI-Compatible Reverse Proxy & Full Request Buffering.

Covers:
- UT_GW_01: OpenAI Chat Completion schema validation, proxy router, headers (REQ-GW-01).
- UT_GW_03: ResponseBuffer lifecycle, token counting, stream buffering (REQ-GW-03, CON-04).
- ST_E2E_01: Official OpenAI Python client SDK compatibility (Sync & Async) with rewrite
  mitigation and header overrides (ADR-0001, ADR-0008).
- ST_STR_01: Full request buffering and zero-token-leakage verification prior to
  mitigation release (ADR-0008, CON-04).
"""

import asyncio
import time
from typing import Any, AsyncIterator, Generator, Iterator, List
import httpx
import openai
from openai.types.chat import ChatCompletion
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.gateway.app import create_app
from src.gateway.buffer import (
    ResponseBuffer,
    abuffer_stream,
    buffer_generator_response,
    buffer_stream,
    buffer_text,
)
from src.gateway.router import (
    format_chat_prompt,
    set_generator_instance,
    set_orchestrator_instance,
)
from src.mitigation.fallback import DEFAULT_REDACTION_TOKEN
from src.mitigation.generator import MockGenerator
from src.retrieval.embedder import MockEmbedder
from src.retrieval.store import DualIndexStore
from src.schemas.openai import ChatMessage
from src.schemas.verification import EvidenceChunk
from src.verifier.nli import MockNLIVerifier
from src.verifier.orchestrator import WaterfallOrchestrator, get_orchestrator

# ============================================================================
# Test Fixtures & Transport Helpers
# ============================================================================


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
    gen = MockGenerator(model_name="mock-gemma-4-E4B", simulated_vram_gb=2.2)
    return gen


@pytest.fixture
def configured_orchestrator(mock_gen: MockGenerator) -> WaterfallOrchestrator:
    """Provide an orchestrator backed by in-memory DualIndexStore and MockNLIVerifier."""
    store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))
    # Seed knowledge into Tier 2 store
    store.add_chunks(
        [
            EvidenceChunk(
                chunk_id="chunk-paris-01",
                text="Paris is the official capital and largest city of France.",
                source_doc="geography_encyclopedia.txt",
                tier=2,
            ),
            EvidenceChunk(
                chunk_id="chunk-physics-01",
                text="Water boils at 100 degrees Celsius under standard atmospheric pressure.",
                source_doc="physics_handbook.txt",
                tier=2,
            ),
            EvidenceChunk(
                chunk_id="chunk-space-01",
                text="The Apollo 11 mission landed on the Moon in July 1969.",
                source_doc="space_history.txt",
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
# UT_GW_01: OpenAI Chat Completion Schema & Reverse Proxy Endpoint Tests
# ============================================================================


def test_ut_gw_01_chat_completions_schema_validation(
    client: TestClient, mock_gen: MockGenerator
) -> None:
    """UT_GW_01: Verify POST /v1/chat/completions accepts valid schemas."""
    mock_gen.register_response(
        "capital of France",
        "Paris is the official capital of France.",
    )

    payload = {
        "model": "google/gemma-4-E4B",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "temperature": 0.0,
        "max_tokens": 128,
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Validate OpenAI ChatCompletion schema invariants
    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert isinstance(data["created"], int)
    assert data["created"] > 0
    assert data["model"] == "google/gemma-4-E4B"

    # Validate Choice structure
    choices = data["choices"]
    assert len(choices) == 1
    choice = choices[0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert "Paris" in choice["message"]["content"]

    # Validate usage metrics
    assert "usage" in data
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0
    assert (
        data["usage"]["total_tokens"]
        == data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
    )

    # Validate verification headers
    assert response.headers["X-Dehallucination-Safe"] == "true"
    assert "X-Hallucination-Rate" in response.headers
    assert "X-Verification-Rate" in response.headers
    assert "X-Needs-Review" in response.headers


def test_ut_gw_01_chat_completions_empty_messages_validation_422(client: TestClient) -> None:
    """UT_GW_01: Verify empty or missing messages payload triggers 422 Unprocessable Entity."""
    # Empty messages list
    res_empty = client.post("/v1/chat/completions", json={"messages": []})
    assert res_empty.status_code == 422

    # Missing messages field
    res_missing = client.post("/v1/chat/completions", json={"model": "google/gemma-4-E4B"})
    assert res_missing.status_code == 422

    # Invalid message item (missing content)
    res_invalid_item = client.post("/v1/chat/completions", json={"messages": [{"role": "user"}]})
    assert res_invalid_item.status_code == 422


def test_ut_gw_01_chat_completions_extra_parameters_accepted(
    client: TestClient, mock_gen: MockGenerator
) -> None:
    """UT_GW_01: Verify extra OpenAI fields are accepted via extra='allow'."""
    mock_gen.register_response("test extra params", "Parameter compatibility verified.")

    payload = {
        "model": "google/gemma-4-E4B",
        "messages": [{"role": "user", "content": "test extra params"}],
        "top_p": 0.95,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.1,
        "n": 1,
        "user": "pes_student_agent",
        "seed": 42,
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert (
        "Parameter compatibility verified." in response.json()["choices"][0]["message"]["content"]
    )


def test_ut_gw_01_chat_completions_verification_headers(
    client: TestClient, mock_gen: MockGenerator
) -> None:
    """UT_GW_01: Verify injection of X-Dehallucination-Safe and rate headers."""
    mock_gen.register_response("greeting", "Hello! Good morning.")

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "greeting"}]},
    )
    assert response.status_code == 200
    assert response.headers.get("X-Dehallucination-Safe") == "true"
    assert response.headers.get("X-Hallucination-Rate") == "0.00"
    assert response.headers.get("X-Needs-Review") in ("true", "false")


def test_ut_gw_01_format_chat_prompt_variations() -> None:
    """UT_GW_01: Verify prompt formatting across single and multi-turn inputs."""
    # 1. Single user message
    msgs_single = [ChatMessage(role="user", content="What is quantum computing?")]
    prompt_str, user_q = format_chat_prompt(msgs_single)
    assert prompt_str == "What is quantum computing?"
    assert user_q == "What is quantum computing?"

    # 2. Multi-turn conversation
    msgs_multi = [
        ChatMessage(role="system", content="You are a physics professor."),
        ChatMessage(role="user", content="Explain gravity."),
        ChatMessage(role="assistant", content="Gravity is a curvature of spacetime."),
        ChatMessage(role="user", content="How does mass affect it?"),
    ]
    prompt_multi, query_multi = format_chat_prompt(msgs_multi)
    assert "System: You are a physics professor." in prompt_multi
    assert "User: Explain gravity." in prompt_multi
    assert "Assistant: Gravity is a curvature of spacetime." in prompt_multi
    assert "User: How does mass affect it?" in prompt_multi
    assert query_multi == "How does mass affect it?"

    # 3. Empty messages
    p_empty, q_empty = format_chat_prompt([])
    assert p_empty == ""
    assert q_empty == ""


def test_ut_gw_01_chat_completions_stream_flag_staged_501(client: TestClient) -> None:
    """UT_GW_01: Verify stream: true returns 501 Not Implemented staging for Ticket 11."""
    payload = {
        "model": "google/gemma-4-E4B",
        "messages": [{"role": "user", "content": "Stream this"}],
        "stream": True,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 501
    assert "Ticket 11" in response.json()["detail"]


# ============================================================================
# UT_GW_03: Full Response Buffering Unit Tests (src/gateway/buffer.py)
# ============================================================================


def test_ut_gw_03_response_buffer_lifecycle_and_properties() -> None:
    """UT_GW_03: Test ResponseBuffer initialization, appending, completion, and immutability."""
    buf = ResponseBuffer()
    assert buf.is_empty is True
    assert buf.is_complete is False
    assert buf.chunk_count == 0
    assert buf.character_count == 0
    assert buf.token_count == 0
    assert buf.content == ""

    # Append chunks
    buf.append("The ")
    buf.append("sky ")
    buf.append("is ")
    buf.append("blue.")

    assert buf.is_empty is False
    assert buf.chunk_count == 4
    assert buf.character_count == len("The sky is blue.")
    assert buf.content == "The sky is blue."
    assert buf.token_count == 4
    assert buf.chunks == ["The ", "sky ", "is ", "blue."]
    assert buf.get_content() == "The sky is blue."
    assert buf.elapsed_ms >= 0.0

    # Seal buffer
    buf.complete()
    assert buf.is_complete is True

    # Appending after completion must raise RuntimeError
    with pytest.raises(RuntimeError, match="Cannot append to a completed ResponseBuffer"):
        buf.append(" Extra text")

    # Clear buffer
    buf.clear()
    assert buf.is_empty is True
    assert buf.is_complete is False
    assert buf.chunk_count == 0


def test_ut_gw_03_buffer_text_helper() -> None:
    """UT_GW_03: Test buffer_text wraps static text in a completed buffer."""
    buf = buffer_text("Sample complete sentence.")
    assert buf.is_complete is True
    assert buf.content == "Sample complete sentence."
    assert buf.chunk_count == 1
    assert buf.token_count == 3

    # Empty text
    buf_empty = buffer_text("")
    assert buf_empty.is_complete is True
    assert buf_empty.content == ""
    assert buf_empty.chunk_count == 0


def test_ut_gw_03_buffer_stream_sync() -> None:
    """UT_GW_03: Test buffer_stream consumes 100% of an iterable chunk stream."""

    def _chunk_generator() -> Iterator[str]:
        yield "Dehallucination "
        yield "add-on "
        yield "module "
        yield "active."

    buf = buffer_stream(_chunk_generator())
    assert buf.is_complete is True
    assert buf.chunk_count == 4
    assert buf.content == "Dehallucination add-on module active."
    assert buf.token_count == 4


@pytest.mark.anyio
async def test_ut_gw_03_abuffer_stream_async() -> None:
    """UT_GW_03: Test abuffer_stream consumes 100% of an asynchronous chunk stream."""

    async def _async_chunk_generator() -> AsyncIterator[str]:
        for token in ["Async ", "stream ", "buffering ", "verified."]:
            await asyncio.sleep(0.001)
            yield token

    buf = await abuffer_stream(_async_chunk_generator())
    assert buf.is_complete is True
    assert buf.chunk_count == 4
    assert buf.content == "Async stream buffering verified."
    assert buf.token_count == 4


def test_ut_gw_03_buffer_generator_response_with_streaming_and_non_streaming() -> None:
    """UT_GW_03: Verify buffer_generator_response consumes stream or static generation."""
    # 1. Generator with generate_stream
    gen = MockGenerator()
    gen.register_response("stream test", "This is a streaming test.")
    buf_stream = buffer_generator_response(gen, prompt="stream test")
    assert buf_stream.is_complete is True
    assert buf_stream.content == "This is a streaming test."
    assert buf_stream.chunk_count > 1

    # 2. Generator without generate_stream (pure static generate)
    class NonStreamingGenerator:
        def generate(self, prompt: str, **kwargs: Any) -> str:
            return "Pure static text"

    static_gen = NonStreamingGenerator()
    buf_static = buffer_generator_response(static_gen, prompt="any")
    assert buf_static.is_complete is True
    assert buf_static.content == "Pure static text"
    assert buf_static.chunk_count == 1


# ============================================================================
# ST_E2E_01: Official OpenAI Python Client SDK Compatibility & Mitigations
# ============================================================================


def test_st_e2e_01_official_openai_sdk_sync_chat_completions(
    client: TestClient, mock_gen: MockGenerator
) -> None:
    """ST_E2E_01: Verify sync openai.OpenAI SDK client executes chat completions."""
    mock_gen.register_response(
        "boiling point",
        "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
    )

    transport = SyncTestClientTransport(client)
    sync_http_client = httpx.Client(transport=transport, base_url="http://testserver/v1")
    openai_client = openai.OpenAI(
        base_url="http://testserver/v1",
        api_key="test-secret-key",
        http_client=sync_http_client,  # type: ignore[arg-type]
    )

    completion = openai_client.chat.completions.create(
        model="google/gemma-4-E4B",
        messages=[{"role": "user", "content": "What is the boiling point of water?"}],
        temperature=0.0,
        max_tokens=64,
    )

    # Validate official OpenAI response class instance
    assert isinstance(completion, ChatCompletion)
    assert completion.model == "google/gemma-4-E4B"
    assert completion.choices[0].finish_reason == "stop"
    assert "100 degrees Celsius" in (completion.choices[0].message.content or "")


@pytest.mark.anyio
async def test_st_e2e_01_official_openai_sdk_async_chat_completions(
    app: FastAPI, mock_gen: MockGenerator
) -> None:
    """ST_E2E_01: Verify official async openai.AsyncOpenAI SDK client executes chat completions."""
    mock_gen.register_response(
        "Apollo 11",
        "The Apollo 11 mission landed on the Moon in July 1969.",
    )

    async_client = openai.AsyncOpenAI(
        base_url="http://testserver/v1",
        api_key="test-secret-key",
        http_client=httpx.AsyncClient(  # type: ignore[arg-type]
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver/v1",
        ),
    )

    completion = await async_client.chat.completions.create(
        model="google/gemma-4-E4B",
        messages=[{"role": "user", "content": "When did Apollo 11 land on the Moon?"}],
        temperature=0.0,
        max_tokens=64,
    )

    assert isinstance(completion, ChatCompletion)
    assert "1969" in (completion.choices[0].message.content or "")


def test_st_e2e_01_default_mitigation_mode_is_rewrite(
    client: TestClient, mock_gen: MockGenerator
) -> None:
    """ST_E2E_01: Verify proxy defaults to mitigation_mode='rewrite'."""
    # Prompt yields hallucinated claim contradicted by knowledge base:
    # KB premise: "Paris is the official capital and largest city of France."
    # Raw generated hallucination: "Berlin is the official capital of France."
    mock_gen.register_response(
        "capital of France",
        "Berlin is the official capital of France.",
    )

    payload = {
        "model": "google/gemma-4-E4B",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Rewrite mitigation should have replaced 'Berlin' with 'Paris'
    sanitized_text = data["choices"][0]["message"]["content"]
    assert "Paris" in sanitized_text
    assert "Berlin" not in sanitized_text
    assert response.headers["X-Dehallucination-Safe"] == "true"


def test_st_e2e_01_header_override_mitigation_flag(
    client: TestClient, mock_gen: MockGenerator
) -> None:
    """ST_E2E_01: Verify X-Mitigation-Mode: flag preserves raw contradicted claim."""
    mock_gen.register_response(
        "capital of France",
        "Berlin is the official capital of France.",
    )

    payload = {
        "model": "google/gemma-4-E4B",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    }

    # Pass X-Mitigation-Mode: flag header
    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Mitigation-Mode": "flag"},
    )
    assert response.status_code == 200
    data = response.json()

    # Raw contradiction is preserved under flag mode
    sanitized_text = data["choices"][0]["message"]["content"]
    assert "Berlin is the official capital of France." in sanitized_text
    assert float(response.headers["X-Hallucination-Rate"]) > 0.0


def test_st_e2e_01_header_override_mitigation_redact(
    client: TestClient, mock_gen: MockGenerator
) -> None:
    """ST_E2E_01: Verify X-Mitigation-Mode: redact masks contradicted claims."""
    mock_gen.register_response(
        "capital of France",
        "Berlin is the official capital of France.",
    )

    payload = {
        "model": "google/gemma-4-E4B",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    }

    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Mitigation-Mode": "redact"},
    )
    assert response.status_code == 200
    data = response.json()

    sanitized_text = data["choices"][0]["message"]["content"]
    assert DEFAULT_REDACTION_TOKEN in sanitized_text
    assert "Berlin" not in sanitized_text


def test_st_e2e_01_header_override_mitigation_block(
    client: TestClient, mock_gen: MockGenerator
) -> None:
    """ST_E2E_01: Verify X-Mitigation-Mode: block raises 422 on contradiction."""
    mock_gen.register_response(
        "capital of France",
        "Berlin is the official capital of France.",
    )

    payload = {
        "model": "google/gemma-4-E4B",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    }

    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Mitigation-Mode": "block"},
    )
    assert response.status_code == 422
    assert "factually grounded response" in response.json()["detail"].lower()


def test_st_e2e_01_header_override_max_tier_and_capping(
    client: TestClient, mock_gen: MockGenerator
) -> None:
    """ST_E2E_01: Verify X-Max-Tier restricts search tiers or caps at ceiling."""
    # Claim only verifiable in Tier 2, not in Tier 1
    mock_gen.register_response(
        "capital of France",
        "Paris is the official capital of France.",
    )

    payload = {
        "model": "google/gemma-4-E4B",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    }

    # Restrict to Tier 1 only (no context docs provided -> claim becomes UNVERIFIED)
    res_t1 = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Max-Tier": "1"},
    )
    assert res_t1.status_code == 200
    assert res_t1.headers["X-Needs-Review"] == "true"  # unverified claim flag

    # Invalid tier header value -> 422
    res_invalid = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Max-Tier": "5"},
    )
    assert res_invalid.status_code == 422


# ============================================================================
# ST_STR_01: Full Request Buffering & Zero-Token-Leakage System Test
# ============================================================================


def test_st_str_01_zero_token_leakage_during_buffering(client: TestClient) -> None:
    """ST_STR_01: Verify that 100% of tokens are buffered before release."""
    generation_timestamps: List[float] = []

    class StreamingTestGenerator(MockGenerator):
        def generate_stream(
            self,
            prompt: str,
            max_new_tokens: int = 512,
            temperature: float = 0.0,
            **kwargs: Any,
        ) -> Iterator[str]:
            for token in ["Berlin ", "is ", "the ", "capital ", "of ", "France."]:
                generation_timestamps.append(time.perf_counter())
                time.sleep(0.005)
                yield token

    set_generator_instance(StreamingTestGenerator())

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "What is the capital of France?"}]},
    )
    assert response.status_code == 200
    data = response.json()

    # Verify that generator emitted all 6 chunks
    assert len(generation_timestamps) == 6

    # Verify that raw hallucination ('Berlin') was fully intercepted and rewritten to 'Paris'
    sanitized_text = data["choices"][0]["message"]["content"]
    assert "Paris" in sanitized_text
    assert "Berlin" not in sanitized_text


def test_st_str_01_buffering_with_in_context_documents(
    client: TestClient, mock_gen: MockGenerator
) -> None:
    """ST_STR_01: Verify buffering and verification against Tier 1 context_documents."""
    context_passage = "Quantum supremacy was achieved in 2019 using the Sycamore processor."
    mock_gen.register_response(
        "quantum supremacy",
        "Quantum supremacy was achieved in 2019 using the Sycamore processor.",
    )

    payload = {
        "model": "google/gemma-4-E4B",
        "messages": [{"role": "user", "content": "When was quantum supremacy achieved?"}],
        "context_documents": [context_passage],
        "max_retrieval_tier": 1,
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "Sycamore" in data["choices"][0]["message"]["content"]
    assert response.headers["X-Dehallucination-Safe"] == "true"
    assert response.headers["X-Verification-Rate"] == "100.00"
