"""Retrieval and knowledge ingestion subsystem (Module 3)."""

from src.retrieval.chunker import chunk_text
from src.retrieval.embedder import BaseEmbedder, FastEmbedEmbedder, MockEmbedder
from src.retrieval.ingest import IngestionPipeline
from src.retrieval.parser import (
    compute_content_hash,
    compute_file_hash,
    extract_text_from_bytes,
    extract_text_from_file,
)
from src.retrieval.store import DualIndexStore

__all__ = [
    "BaseEmbedder",
    "DualIndexStore",
    "FastEmbedEmbedder",
    "IngestionPipeline",
    "MockEmbedder",
    "chunk_text",
    "compute_content_hash",
    "compute_file_hash",
    "extract_text_from_bytes",
    "extract_text_from_file",
]
