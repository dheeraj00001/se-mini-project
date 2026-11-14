"""Retrieval adapters and rank fusion for multi-tier grounding.

Implements REQ-RET-04, REQ-RET-05, and ADR-0004:
- Tier 1: In-context request document retrieval (lexical BM25 and/or dense embedding).
- Tier 2: Local hybrid knowledge base retrieval (concurrent Qdrant dense + BM25 sparse)
  merged via Reciprocal Rank Fusion (k=60).
- Unified RetrievalAdapter interface and adapter factory for waterfall escalation.
"""

from abc import ABC, abstractmethod
import concurrent.futures
from typing import Any, List, Optional, Sequence, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.chunker import chunk_text
from src.retrieval.embedder import BaseEmbedder
from src.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from src.retrieval.store import DualIndexStore
from src.schemas.verification import EvidenceCandidate, EvidenceChunk


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two numeric vectors."""
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class BaseRetrievalAdapter(ABC):
    """Abstract base contract for tier-specific retrieval adapters."""

    @abstractmethod
    def search(self, claim: str, limit: int = 5, **kwargs: Any) -> List[EvidenceChunk]:
        """Search evidence passages given an atomic claim proposition.

        Args:
            claim: Atomic factual claim string to verify.
            limit: Maximum number of candidate chunks to return.
            **kwargs: Additional tier-specific options.

        Returns:
            Ranked list of EvidenceChunk candidates.
        """
        pass


class Tier1ContextAdapter(BaseRetrievalAdapter):
    """Tier 1: Retrieves candidate passages from in-context request documents."""

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        context_documents: Optional[Sequence[str]] = None,
    ) -> None:
        self._embedder = embedder
        self._default_context_documents = list(context_documents) if context_documents else []

    def search(
        self,
        claim: str,
        limit: int = 5,
        context_documents: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[EvidenceChunk]:
        """Extract and score relevant passages from in-context request documents.

        Args:
            claim: Standalone atomic claim query.
            limit: Maximum candidate passages to return.
            context_documents: Optional list of raw document texts from the request.
                If omitted, uses default context documents provided at init.

        Returns:
            Ranked candidate EvidenceChunk instances with tier=1.
        """
        cleaned_claim = claim.strip()
        if not cleaned_claim:
            return []

        docs = (
            list(context_documents)
            if context_documents is not None
            else self._default_context_documents
        )
        if not docs:
            return []

        # 1. Segment incoming context documents into valid chunks
        chunks: List[EvidenceChunk] = []
        for idx, doc_text in enumerate(docs):
            cleaned_doc = doc_text.strip()
            if not cleaned_doc:
                continue
            doc_id = f"context_doc_{idx + 1}"
            doc_chunks = chunk_text(
                text=cleaned_doc,
                source_doc=doc_id,
                window_size=250,
                overlap=50,
                tier=1,
            )
            chunks.extend(doc_chunks)

        if not chunks:
            return []

        # 2. Compute BM25 sparse scores
        tokenized_corpus = [c.text.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = cleaned_claim.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        query_words = set(tokenized_query)

        sparse_candidates = []
        for chunk, score in zip(chunks, bm25_scores):
            overlap = query_words.intersection(chunk.text.lower().split())
            if score > 0.0 or overlap:
                eff_score = float(score) if score > 0.0 else float(len(overlap))
                sparse_candidates.append((chunk, eff_score))

        sparse_candidates.sort(key=lambda x: (-x[1], x[0].chunk_id))

        # 3. Dense embeddings (if embedder provided)
        dense_candidates: List[Tuple[EvidenceChunk, float]] = []
        if self._embedder is not None:
            query_vec = self._embedder.embed_query(cleaned_claim)
            chunk_texts = [c.text for c in chunks]
            doc_vecs = self._embedder.embed_documents(chunk_texts)
            for chunk, d_vec in zip(chunks, doc_vecs):
                sim = _cosine_similarity(query_vec, d_vec)
                dense_candidates.append((chunk, sim))
            dense_candidates.sort(key=lambda x: (-x[1], x[0].chunk_id))

        # 4. Rank aggregation: combine dense + sparse if both available, else best available
        if dense_candidates and sparse_candidates:
            fused = reciprocal_rank_fusion(
                [dense_candidates, sparse_candidates],
                k=DEFAULT_RRF_K,
                limit=limit,
            )
            # Ensure tier is explicitly 1
            return [c.model_copy(update={"tier": 1}) for c in fused]
        elif dense_candidates:
            selected = dense_candidates[:limit]
            return [c.model_copy(update={"score": score, "tier": 1}) for c, score in selected]
        elif sparse_candidates:
            selected = sparse_candidates[:limit]
            return [c.model_copy(update={"score": score, "tier": 1}) for c, score in selected]

        # No positive matches
        return []


class Tier2HybridAdapter(BaseRetrievalAdapter):
    """Tier 2: Queries local Qdrant and BM25 concurrently and rank-fuses via RRF."""

    def __init__(self, store: Optional[DualIndexStore] = None) -> None:
        self._store = store if store is not None else DualIndexStore()

    @property
    def store(self) -> DualIndexStore:
        """Return the underlying DualIndexStore."""
        return self._store

    def search(
        self,
        claim: str,
        limit: int = 5,
        dense_limit: int = 10,
        sparse_limit: int = 10,
        k: int = DEFAULT_RRF_K,
        **kwargs: Any,
    ) -> List[EvidenceChunk]:
        """Concurrently query Qdrant (dense) and BM25 (sparse) and fuse with RRF.

        Args:
            claim: Standalone atomic claim query.
            limit: Maximum candidate passages to return (top-K).
            dense_limit: Number of candidates to fetch from Qdrant vector search.
            sparse_limit: Number of candidates to fetch from BM25 sparse search.
            k: Reciprocal Rank Fusion smoothing parameter (default: 60).

        Returns:
            Top-K reranked EvidenceChunk candidates with combined RRF scores.
        """
        cleaned_claim = claim.strip()
        if not cleaned_claim:
            return []

        # Execute dense and sparse retrieval concurrently across thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(self._store.search_dense, cleaned_claim, dense_limit)
            future_sparse = executor.submit(self._store.search_sparse, cleaned_claim, sparse_limit)
            dense_results = future_dense.result()
            sparse_results = future_sparse.result()

        # Combine rankings using RRF
        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=k,
            limit=limit,
        )

        # Enforce tier=2 on all returned chunks
        return [c.model_copy(update={"tier": 2}) for c in fused]


class RetrievalAdapter:
    """Unified retrieval gateway routing requests to Tier 1, 2, or 3 adapters."""

    def __init__(
        self,
        tier1_adapter: Optional[Tier1ContextAdapter] = None,
        tier2_adapter: Optional[Tier2HybridAdapter] = None,
        store: Optional[DualIndexStore] = None,
        embedder: Optional[BaseEmbedder] = None,
    ) -> None:
        self._tier1 = (
            tier1_adapter if tier1_adapter is not None else Tier1ContextAdapter(embedder=embedder)
        )
        self._tier2 = (
            tier2_adapter if tier2_adapter is not None else Tier2HybridAdapter(store=store)
        )

    @property
    def tier1_adapter(self) -> Tier1ContextAdapter:
        """Return the Tier 1 Context Adapter."""
        return self._tier1

    @property
    def tier2_adapter(self) -> Tier2HybridAdapter:
        """Return the Tier 2 Hybrid Adapter."""
        return self._tier2

    def search(
        self,
        claim: str,
        tier: int = 2,
        context_docs: Optional[Sequence[str]] = None,
        limit: int = 5,
        **kwargs: Any,
    ) -> List[EvidenceCandidate]:
        """Search evidence candidates across the specified grounding tier.

        Args:
            claim: Atomic factual claim string.
            tier: Grounding tier (1=Context, 2=Local KB, 3=Web).
            context_docs: Tier 1 in-context document passages.
            limit: Maximum candidate passages to return.

        Returns:
            Ranked list of EvidenceCandidate instances with scores.

        Raises:
            ValueError: If tier is not in [1, 2, 3].
        """
        if tier not in (1, 2, 3):
            raise ValueError(f"Invalid retrieval tier: {tier}. Must be between 1 and 3.")

        if tier == 1:
            return self._tier1.search(
                claim=claim,
                limit=limit,
                context_documents=context_docs,
                **kwargs,
            )
        elif tier == 2:
            return self._tier2.search(
                claim=claim,
                limit=limit,
                **kwargs,
            )
        else:
            # Tier 3 (External Web Search) is implemented in Sprint 2 (Ticket 09)
            return []


def get_retrieval_adapter(
    store: Optional[DualIndexStore] = None,
    embedder: Optional[BaseEmbedder] = None,
) -> RetrievalAdapter:
    """Factory helper to instantiate a configured RetrievalAdapter."""
    return RetrievalAdapter(store=store, embedder=embedder)
