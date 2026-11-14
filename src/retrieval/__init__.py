"""Retrieval and knowledge ingestion subsystem (Module 3)."""

from src.retrieval.adapters import (
    BaseRetrievalAdapter,
    RetrievalAdapter,
    Tier1ContextAdapter,
    Tier2HybridAdapter,
    get_retrieval_adapter,
)
from src.retrieval.chunker import chunk_text
from src.retrieval.embedder import (
    BaseEmbedder,
    EmbeddingGemmaEmbedder,
    FastEmbedEmbedder,
    MockEmbedder,
    get_default_embedder,
)
from src.retrieval.fusion import (
    DEFAULT_RRF_K,
    compute_rrf_score,
    reciprocal_rank_fusion,
)
from src.retrieval.ingest import IngestionPipeline
from src.retrieval.parser import (
    compute_content_hash,
    compute_file_hash,
    extract_text_from_bytes,
    extract_text_from_file,
)
from src.retrieval.store import DualIndexStore
from src.schemas.verification import EvidenceCandidate, EvidenceChunk

__all__ = [
    "BaseEmbedder",
    "BaseRetrievalAdapter",
    "DEFAULT_RRF_K",
    "DualIndexStore",
    "EmbeddingGemmaEmbedder",
    "EvidenceCandidate",
    "EvidenceChunk",
    "FastEmbedEmbedder",
    "IngestionPipeline",
    "MockEmbedder",
    "RetrievalAdapter",
    "Tier1ContextAdapter",
    "Tier2HybridAdapter",
    "chunk_text",
    "compute_content_hash",
    "compute_file_hash",
    "compute_rrf_score",
    "extract_text_from_bytes",
    "extract_text_from_file",
    "get_default_embedder",
    "get_retrieval_adapter",
    "reciprocal_rank_fusion",
]
