"""Pydantic schemas for OpenAI-compatible reverse proxy endpoints (REQ-GW-01).

Implements standard OpenAI Chat Completion request and response schemas,
allowing drop-in replacement for OpenAI SDK clients and tools.
"""

import time
import uuid
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.schemas.verification import MitigationMode


class ChatMessage(BaseModel):
    """Individual chat message within an OpenAI chat completion payload."""

    model_config = ConfigDict(extra="allow")

    role: str = Field(
        ...,
        description="Role of the message author: 'system', 'user', or 'assistant'.",
    )
    content: str = Field(
        ...,
        description="Text content of the message.",
    )
    name: Optional[str] = Field(
        default=None,
        description="Optional author name for multi-party conversations.",
    )


class ChatCompletionRequest(BaseModel):
    """Request payload for POST /v1/chat/completions (REQ-GW-01)."""

    model_config = ConfigDict(extra="allow")

    messages: List[ChatMessage] = Field(
        ...,
        min_length=1,
        description="Chronological list of chat messages comprising the conversation.",
    )
    model: str = Field(
        default="google/gemma-4-E4B",
        description="ID of the model to query.",
    )
    temperature: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for text generation.",
    )
    max_tokens: Optional[int] = Field(
        default=512,
        ge=1,
        description="Maximum number of tokens to generate.",
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Whether to stream response chunks via Server-Sent Events (ADR-0008).",
    )
    context_documents: Optional[List[str]] = Field(
        default_factory=list,
        description="Optional Tier 1 in-context reference passages for grounding.",
    )
    max_retrieval_tier: Optional[int] = Field(
        default=None,
        ge=1,
        le=3,
        description="Optional caller override for maximum retrieval tier (1, 2, or 3).",
    )
    mitigation_mode: Optional[MitigationMode] = Field(
        default=None,
        description="Optional caller override for mitigation action: flag, rewrite, redact, block.",
    )


class ChatCompletionMessage(BaseModel):
    """Assistant response message within an OpenAI ChatCompletion choice."""

    model_config = ConfigDict(extra="allow")

    role: str = Field(
        default="assistant",
        description="Role of the assistant message author.",
    )
    content: str = Field(
        ...,
        description="Verified and sanitized text content.",
    )


class ChatCompletionChoice(BaseModel):
    """Individual completion outcome choice."""

    model_config = ConfigDict(extra="allow")

    index: int = Field(
        default=0,
        description="Choice sequence index.",
    )
    message: ChatCompletionMessage = Field(
        ...,
        description="Assistant completion message.",
    )
    finish_reason: str = Field(
        default="stop",
        description="Reason generation terminated: 'stop', 'length', etc.",
    )


class ChatCompletionUsage(BaseModel):
    """Token consumption statistics for the completion."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = Field(
        default=0,
        ge=0,
        description="Estimated token count of the prompt.",
    )
    completion_tokens: int = Field(
        default=0,
        ge=0,
        description="Estimated token count of the generated response.",
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total token count consumed.",
    )


class ChatCompletionResponse(BaseModel):
    """Complete OpenAI-compatible ChatCompletion JSON response object."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(
        default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}",
        description="Unique completion identifier.",
    )
    object: str = Field(
        default="chat.completion",
        description="Object type descriptor.",
    )
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="Unix epoch timestamp in seconds.",
    )
    model: str = Field(
        ...,
        description="Model identifier that produced the completion.",
    )
    choices: List[ChatCompletionChoice] = Field(
        ...,
        min_length=1,
        description="List of completion outcome choices.",
    )
    usage: Optional[ChatCompletionUsage] = Field(
        default=None,
        description="Optional token usage metrics.",
    )
