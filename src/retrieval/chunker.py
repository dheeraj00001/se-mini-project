"""Deterministic text chunking and SHA-256 ID calculation.

Implements REQ-RET-01 and REQ-RET-02:
- Sliding token window of 250-300 tokens with 50-token overlap.
- Deterministic SHA-256 chunk IDs derived from source metadata and passage text.
"""

import hashlib
from typing import List

from src.schemas.verification import EvidenceChunk


def chunk_text(
    text: str,
    source_doc: str,
    window_size: int = 250,
    overlap: int = 50,
    tier: int = 2,
) -> List[EvidenceChunk]:
    """Segment a continuous text into sliding token windows with deterministic SHA-256 IDs.

    Args:
        text: Continuous raw document or passage text.
        source_doc: Document identifier or relative path.
        window_size: Maximum token count per chunk (standard: 250-300 tokens).
        overlap: Token overlap between adjacent chunks (standard: 50 tokens).
        tier: Grounding tier (default: 2 for local knowledge base).

    Returns:
        List of typed EvidenceChunk instances with unique SHA-256 chunk IDs.

    Raises:
        ValueError: If window_size <= 0, overlap < 0, or overlap >= window_size.
    """
    if window_size <= 0:
        raise ValueError("window_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= window_size:
        raise ValueError("window_size must be greater than overlap")

    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    tokens = cleaned_text.split()
    total_tokens = len(tokens)
    if total_tokens == 0:
        return []

    if total_tokens <= window_size:
        chunk_str = " ".join(tokens)
        chunk_id = hashlib.sha256(f"{source_doc}:0:{chunk_str}".encode("utf-8")).hexdigest()

        return [
            EvidenceChunk(
                chunk_id=chunk_id,
                text=chunk_str,
                source_doc=source_doc,
                tier=tier,
                score=None,
                metadata={
                    "source_doc": source_doc,
                    "chunk_index": 0,
                    "token_count": total_tokens,
                    "start_token": 0,
                    "end_token": total_tokens,
                },
            )
        ]

    step = window_size - overlap
    chunks: List[EvidenceChunk] = []
    start = 0
    chunk_index = 0

    while start < total_tokens:
        end = min(start + window_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_str = " ".join(chunk_tokens)

        chunk_id = hashlib.sha256(
            f"{source_doc}:{chunk_index}:{chunk_str}".encode("utf-8")
        ).hexdigest()

        chunks.append(
            EvidenceChunk(
                chunk_id=chunk_id,
                text=chunk_str,
                source_doc=source_doc,
                tier=tier,
                score=None,
                metadata={
                    "source_doc": source_doc,
                    "chunk_index": chunk_index,
                    "token_count": len(chunk_tokens),
                    "start_token": start,
                    "end_token": end,
                },
            )
        )

        chunk_index += 1
        if end >= total_tokens:
            break
        start += step

    return chunks
