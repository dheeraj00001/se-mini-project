"""Synthetic SSE Replay Streaming Engine for OpenAI-compatible gateway (REQ-GW-04, ADR-0008).

Splits verified and sanitized output into progressive token chunks and streams them
over Server-Sent Events (SSE) with realistic pacing (~10-20 ms per token) to provide
drop-in compatibility with OpenAI streaming clients while ensuring zero hallucination leakage.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union
from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.schemas.openai import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
)

logger = logging.getLogger(__name__)

# Default synthetic pacing delay (~15ms per token per ADR-0008)
DEFAULT_PACING_DELAY_MS: float = float(os.environ.get("DEHALLUCINATION_SSE_PACING_MS", "15.0"))
_current_pacing_delay_ms: float = DEFAULT_PACING_DELAY_MS


def get_sse_pacing_delay_ms() -> float:
    """Retrieve global default pacing delay in milliseconds."""
    return _current_pacing_delay_ms


def set_sse_pacing_delay_ms(delay_ms: float) -> None:
    """Set global default pacing delay in milliseconds (useful for tests)."""
    global _current_pacing_delay_ms
    _current_pacing_delay_ms = max(0.0, float(delay_ms))


def split_text_into_tokens(text: str) -> List[str]:
    """Split text into word/token chunks preserving all whitespace and punctuation.

    Args:
        text: Input string to segment into tokens.

    Returns:
        List of string tokens such that ''.join(tokens) == text.
    """
    if not text:
        return []
    # Match non-whitespace sequences optionally followed by trailing whitespace,
    # or standalone whitespace runs.
    token_pattern = re.compile(r"\S+\s*|\s+")
    tokens = token_pattern.findall(text)
    return tokens


def format_sse_event(
    data: Union[str, BaseModel, Dict[str, Any]],
    event: Optional[str] = None,
) -> str:
    """Format data payload as a Server-Sent Event (SSE) packet.

    Args:
        data: The payload to serialize. Can be '[DONE]', a Pydantic model, a dict, or str.
        event: Optional event type name.

    Returns:
        Formatted SSE string terminated by double newlines: 'data: ...\\n\\n'.
    """
    if data == "[DONE]":
        payload = "[DONE]"
    elif isinstance(data, BaseModel):
        payload = data.model_dump_json()
    elif isinstance(data, (dict, list)):
        payload = json.dumps(data)
    else:
        payload = str(data)

    if event:
        return f"event: {event}\ndata: {payload}\n\n"
    return f"data: {payload}\n\n"


class SyntheticSSEReplay:
    """Synthetic Server-Sent Event (SSE) Replay Engine (REQ-GW-04, ADR-0008).

    Emits verified tokens as OpenAI-compatible streaming delta chunks with
    configurable pacing delay and disconnection awareness.
    """

    def __init__(self, pacing_delay_ms: Optional[float] = None) -> None:
        """Initialize replay engine with specified or default pacing delay."""
        if pacing_delay_ms is not None:
            self.pacing_delay_ms = max(0.0, float(pacing_delay_ms))
        else:
            self.pacing_delay_ms = get_sse_pacing_delay_ms()

    def generate_chunks(
        self,
        text: str,
        model: str,
        completion_id: Optional[str] = None,
        created_ts: Optional[int] = None,
    ) -> Iterator[ChatCompletionChunk]:
        """Generate chronological ChatCompletionChunk objects for the given text.

        Sequence:
        1. Initial chunk with delta.role = 'assistant' and empty content.
        2. Content chunks for each token in text.
        3. Terminal chunk with finish_reason = 'stop'.
        """
        cid = completion_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
        cts = created_ts if created_ts is not None else int(time.time())

        # 1. Initial role declaration chunk
        yield ChatCompletionChunk(
            id=cid,
            object="chat.completion.chunk",
            created=cts,
            model=model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(role="assistant", content=""),
                    finish_reason=None,
                )
            ],
        )

        # 2. Content delta chunks
        tokens = split_text_into_tokens(text)
        for token in tokens:
            yield ChatCompletionChunk(
                id=cid,
                object="chat.completion.chunk",
                created=cts,
                model=model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=token),
                        finish_reason=None,
                    )
                ],
            )

        # 3. Final stop chunk
        yield ChatCompletionChunk(
            id=cid,
            object="chat.completion.chunk",
            created=cts,
            model=model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(),
                    finish_reason="stop",
                )
            ],
        )

    def iter_sse_events(
        self,
        text: str,
        model: str,
        completion_id: Optional[str] = None,
        created_ts: Optional[int] = None,
    ) -> Iterator[str]:
        """Synchronously yield formatted SSE event strings ending with data: [DONE]."""
        for chunk in self.generate_chunks(
            text=text,
            model=model,
            completion_id=completion_id,
            created_ts=created_ts,
        ):
            yield format_sse_event(chunk)
        yield format_sse_event("[DONE]")

    async def aiter_sse_events(
        self,
        text: str,
        model: str,
        completion_id: Optional[str] = None,
        created_ts: Optional[int] = None,
        request: Optional[Request] = None,
    ) -> AsyncIterator[str]:
        """Asynchronously yield formatted SSE event strings with paced delays.

        Monitors connection state via request.is_disconnected() to prevent orphaned work.
        """
        delay_sec = self.pacing_delay_ms / 1000.0

        try:
            for chunk in self.generate_chunks(
                text=text,
                model=model,
                completion_id=completion_id,
                created_ts=created_ts,
            ):
                if request is not None and await request.is_disconnected():
                    logger.info("Client disconnected during SSE replay; aborting stream.")
                    return

                yield format_sse_event(chunk)

                if delay_sec > 0:
                    await asyncio.sleep(delay_sec)

            if request is not None and await request.is_disconnected():
                return

            yield format_sse_event("[DONE]")

        except asyncio.CancelledError:
            logger.info("SSE streaming task cancelled by client disconnect.")
            raise


def build_sse_streaming_response(
    text: str,
    model: str,
    verification_headers: Optional[Dict[str, str]] = None,
    pacing_delay_ms: Optional[float] = None,
    request: Optional[Request] = None,
    completion_id: Optional[str] = None,
    created_ts: Optional[int] = None,
) -> StreamingResponse:
    """Construct an ASGI StreamingResponse for synthetic SSE replay.

    Args:
        text: Finalized, verified, and sanitized text to replay.
        model: Identifier of the generating model.
        verification_headers: Metadata headers (e.g. X-Dehallucination-Safe).
        pacing_delay_ms: Optional pacing delay override.
        request: Optional incoming Request for client disconnect detection.
        completion_id: Unique chat completion identifier.
        created_ts: Generation epoch timestamp.

    Returns:
        FastAPI / Starlette StreamingResponse with text/event-stream media type.
    """
    engine = SyntheticSSEReplay(pacing_delay_ms=pacing_delay_ms)
    event_generator = engine.aiter_sse_events(
        text=text,
        model=model,
        completion_id=completion_id,
        created_ts=created_ts,
        request=request,
    )

    headers: Dict[str, str] = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    if verification_headers:
        headers.update(verification_headers)

    return StreamingResponse(
        event_generator,
        media_type="text/event-stream",
        headers=headers,
    )
