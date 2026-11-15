"""Full response buffering engine for reverse proxy and streaming interception.

Implements REQ-GW-03, CON-04, and ADR-0008.
Traceability: UT_GW_03, ST_STR_01.

Core Responsibilities:
1. ResponseBuffer: In-memory token accumulator ensuring 100% of raw model output
   is collected prior to claim segmentation, waterfall retrieval, or NLI verification.
2. Zero Token Leakage: Guarantees that unverified or hallucinated raw tokens never
   reach downstream clients before full document verification completes.
3. Synchronous and Asynchronous Stream Consumption: Efficiently consumes token
   streams from local LLM generators or remote upstream endpoints.
"""

import logging
import time
from typing import Any, AsyncIterable, Iterable, List, Optional

logger = logging.getLogger("gateway.buffer")


class ResponseBuffer:
    """In-memory buffer that captures 100% of raw generated text before verification."""

    def __init__(self) -> None:
        self._chunks: List[str] = []
        self._is_complete: bool = False
        self._start_time: float = time.perf_counter()
        self._end_time: Optional[float] = None

    @property
    def chunks(self) -> List[str]:
        """List of all collected token or text chunks."""
        return list(self._chunks)

    @property
    def chunk_count(self) -> int:
        """Total number of discrete chunks appended to this buffer."""
        return len(self._chunks)

    @property
    def is_complete(self) -> bool:
        """True if buffering has concluded and the buffer is sealed."""
        return self._is_complete

    @property
    def is_empty(self) -> bool:
        """True if no chunks have been appended to this buffer."""
        return len(self._chunks) == 0

    @property
    def content(self) -> str:
        """Consolidated string of all buffered chunks."""
        return "".join(self._chunks)

    @property
    def character_count(self) -> int:
        """Total number of characters in the buffered content."""
        return sum(len(c) for c in self._chunks)

    @property
    def token_count(self) -> int:
        """Estimated token count of the buffered content (whitespace heuristic)."""
        text = self.content.strip()
        if not text:
            return 0
        return len(text.split())

    @property
    def elapsed_ms(self) -> float:
        """Elapsed buffering duration in milliseconds."""
        end = self._end_time if self._end_time is not None else time.perf_counter()
        return (end - self._start_time) * 1000.0

    def append(self, chunk: str) -> None:
        """Append a token or text chunk to the buffer.

        Raises:
            RuntimeError: If attempting to append to an already sealed buffer.
        """
        if self._is_complete:
            raise RuntimeError("Cannot append to a completed ResponseBuffer.")
        if chunk:
            self._chunks.append(chunk)

    def complete(self) -> None:
        """Seal the buffer, marking buffering as 100% complete."""
        if not self._is_complete:
            self._is_complete = True
            self._end_time = time.perf_counter()

    def clear(self) -> None:
        """Reset the buffer state for reuse."""
        self._chunks.clear()
        self._is_complete = False
        self._start_time = time.perf_counter()
        self._end_time = None

    def get_content(self) -> str:
        """Return the complete accumulated content string."""
        return self.content


def buffer_text(text: str) -> ResponseBuffer:
    """Wrap a complete static text string in a sealed ResponseBuffer."""
    buffer = ResponseBuffer()
    if text:
        buffer.append(text)
    buffer.complete()
    return buffer


def buffer_stream(stream: Iterable[str]) -> ResponseBuffer:
    """Consume an iterable stream of token chunks 100% into a ResponseBuffer.

    Implements REQ-GW-03 and CON-04: guarantees all tokens are buffered
    before releasing output to verification.
    """
    buffer = ResponseBuffer()
    for chunk in stream:
        buffer.append(chunk)
    buffer.complete()
    logger.debug(
        "Buffered stream: %d chunks, %d chars, %.2f ms",
        buffer.chunk_count,
        buffer.character_count,
        buffer.elapsed_ms,
    )
    return buffer


async def abuffer_stream(stream: AsyncIterable[str]) -> ResponseBuffer:
    """Asynchronously consume a token chunk stream 100% into a ResponseBuffer."""
    buffer = ResponseBuffer()
    async for chunk in stream:
        buffer.append(chunk)
    buffer.complete()
    logger.debug(
        "Asynchronously buffered stream: %d chunks, %d chars, %.2f ms",
        buffer.chunk_count,
        buffer.character_count,
        buffer.elapsed_ms,
    )
    return buffer


def buffer_generator_response(
    generator: Any,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    **kwargs: Any,
) -> ResponseBuffer:
    """Prompt the generator model and buffer 100% of the raw response (REQ-GW-03).

    Checks whether the generator supports streaming generation; if so,
    consumes the full stream before returning. Otherwise, invokes standard
    generation and packages the text in a sealed buffer.
    """
    if hasattr(generator, "generate_stream") and callable(generator.generate_stream):
        stream = generator.generate_stream(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        return buffer_stream(stream)

    raw_text = generator.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        **kwargs,
    )
    return buffer_text(raw_text)
