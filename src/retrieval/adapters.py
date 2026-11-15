"""Retrieval adapters and rank fusion for multi-tier grounding.

Implements REQ-RET-04, REQ-RET-05, and ADR-0004:
- Tier 1: In-context request document retrieval (lexical BM25 and/or dense embedding).
- Tier 2: Local hybrid knowledge base retrieval (concurrent Qdrant dense + BM25 sparse)
  merged via Reciprocal Rank Fusion (k=60).
- Unified RetrievalAdapter interface and adapter factory for waterfall escalation.
"""

from abc import ABC, abstractmethod
import asyncio
import concurrent.futures
import logging
from typing import Any, List, Optional, Sequence, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

from src.gateway.middleware import get_server_max_tier
from src.retrieval.chunker import chunk_text
from src.retrieval.embedder import BaseEmbedder
from src.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from src.retrieval.parser import compute_content_hash
from src.retrieval.store import DualIndexStore
from src.schemas.verification import EvidenceCandidate, EvidenceChunk

logger = logging.getLogger(__name__)


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


class Tier3WebAdapter(BaseRetrievalAdapter):
    """Tier 3: Queries live web snippets via DuckDuckGo for external grounding.

    Implements REQ-RET-05, REQ-RET-06, NFR-SEC-01, and ADR-0004.
    """

    def __init__(
        self,
        ddgs_client: Optional[Any] = None,
        timeout: float = 5.0,
        server_max_tier: Optional[int] = None,
        region: Optional[str] = None,
        safesearch: str = "moderate",
    ) -> None:
        """Initialize the Tier 3 DuckDuckGo web retrieval adapter.

        Args:
            ddgs_client: Optional DuckDuckGo search client (e.g. DDGS instance or mock).
            timeout: HTTP query timeout in seconds (default: 5.0).
            server_max_tier: Server-side maximum allowed tier ceiling (default: from env or 3).
            region: Optional DuckDuckGo search region string (e.g. 'us-en').
            safesearch: Safesearch level ('on', 'moderate', 'off'; default: 'moderate').
        """
        self._ddgs_client = ddgs_client
        self._timeout = timeout
        self._server_max_tier = (
            server_max_tier if server_max_tier is not None else get_server_max_tier()
        )
        self._region = region
        self._safesearch = safesearch

    @property
    def ddgs_client(self) -> Optional[Any]:
        """Return the underlying DDGS client instance or mock."""
        return self._ddgs_client

    @property
    def timeout(self) -> float:
        """Return the query timeout in seconds."""
        return self._timeout

    @property
    def server_max_tier(self) -> int:
        """Return the immutable server max tier ceiling."""
        return self._server_max_tier

    def search(
        self,
        claim: str,
        limit: int = 5,
        client_max_tier: int = 3,
        **kwargs: Any,
    ) -> List[EvidenceChunk]:
        """Execute DuckDuckGo search for atomic claim proposition.

        Args:
            claim: Standalone atomic claim query string.
            limit: Maximum candidate passages to return (default: 5).
            client_max_tier: Explicit client tier request (must be == 3 to opt in).
            **kwargs: Additional parameters forwarded to search.

        Returns:
            Ranked list of EvidenceChunk instances with tier=3 and URL citations.
        """
        # Strict opt-in security ceiling checks (REQ-RET-05, REQ-RET-06, NFR-SEC-01)
        if client_max_tier != 3:
            logger.debug(
                "Tier 3 search aborted: client requested tier %d (!= 3)",
                client_max_tier,
            )
            return []

        if self._server_max_tier < 3:
            logger.warning(
                "Tier 3 search blocked: server max tier ceiling is %d (< 3)",
                self._server_max_tier,
            )
            return []

        cleaned_claim = claim.strip()
        if not cleaned_claim:
            return []

        try:
            if self._ddgs_client is not None:
                client = self._ddgs_client
            else:
                from duckduckgo_search import DDGS

                client = DDGS(timeout=int(self._timeout))

            raw_results = client.text(
                keywords=cleaned_claim,
                max_results=limit,
                region=self._region,
                safesearch=self._safesearch,
            )
        except Exception as exc:
            # Graceful degradation on rate limits, timeouts, or network disconnects
            logger.warning(
                "DuckDuckGo search error for claim '%s': %s",
                cleaned_claim,
                exc,
            )
            return []

        if not raw_results:
            return []

        chunks: List[EvidenceChunk] = []
        for idx, item in enumerate(raw_results):
            if not isinstance(item, dict):
                continue
            body = str(item.get("body") or item.get("snippet") or "").strip()
            title = str(item.get("title") or "").strip()
            href = str(item.get("href") or item.get("link") or "").strip()

            if body and title:
                snippet_text = f"{title}: {body}"
            elif body:
                snippet_text = body
            elif title:
                snippet_text = title
            else:
                continue

            source_doc = href if href else f"duckduckgo_search_{idx + 1}"
            chunk_id = compute_content_hash(f"{source_doc}:{snippet_text}")
            score = round(1.0 / (idx + 1), 4)

            chunk = EvidenceChunk(
                chunk_id=chunk_id,
                text=snippet_text,
                source_doc=source_doc,
                tier=3,
                score=score,
                metadata={
                    "url": href,
                    "title": title,
                    "source": "duckduckgo",
                    "rank": idx + 1,
                },
            )
            chunks.append(chunk)
            if len(chunks) >= limit:
                break

        return chunks

    async def asearch(
        self,
        claim: str,
        limit: int = 5,
        client_max_tier: int = 3,
        **kwargs: Any,
    ) -> List[EvidenceChunk]:
        """Asynchronously execute DuckDuckGo search on a worker thread."""
        return await asyncio.to_thread(
            self.search,
            claim=claim,
            limit=limit,
            client_max_tier=client_max_tier,
            **kwargs,
        )


class RetrievalAdapter:
    """Unified retrieval gateway routing requests to Tier 1, 2, or 3 adapters."""

    def __init__(
        self,
        tier1_adapter: Optional[Tier1ContextAdapter] = None,
        tier2_adapter: Optional[Tier2HybridAdapter] = None,
        tier3_adapter: Optional[Tier3WebAdapter] = None,
        store: Optional[DualIndexStore] = None,
        embedder: Optional[BaseEmbedder] = None,
        server_max_tier: Optional[int] = None,
    ) -> None:
        self._tier1 = (
            tier1_adapter if tier1_adapter is not None else Tier1ContextAdapter(embedder=embedder)
        )
        self._tier2 = (
            tier2_adapter if tier2_adapter is not None else Tier2HybridAdapter(store=store)
        )
        self._tier3 = (
            tier3_adapter
            if tier3_adapter is not None
            else Tier3WebAdapter(server_max_tier=server_max_tier)
        )

    @property
    def tier1_adapter(self) -> Tier1ContextAdapter:
        """Return the Tier 1 Context Adapter."""
        return self._tier1

    @property
    def tier2_adapter(self) -> Tier2HybridAdapter:
        """Return the Tier 2 Hybrid Adapter."""
        return self._tier2

    @property
    def tier3_adapter(self) -> Tier3WebAdapter:
        """Return the Tier 3 Web Adapter."""
        return self._tier3

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
            client_tier = kwargs.pop("client_max_tier", 3)
            return self._tier3.search(
                claim=claim,
                limit=limit,
                client_max_tier=client_tier,
                **kwargs,
            )

    async def asearch(
        self,
        claim: str,
        tier: int = 2,
        context_docs: Optional[Sequence[str]] = None,
        limit: int = 5,
        **kwargs: Any,
    ) -> List[EvidenceCandidate]:
        """Asynchronously search evidence candidates across the specified tier."""
        if tier not in (1, 2, 3):
            raise ValueError(f"Invalid retrieval tier: {tier}. Must be between 1 and 3.")

        if tier == 3:
            client_tier = kwargs.pop("client_max_tier", 3)
            return await self._tier3.asearch(
                claim=claim,
                limit=limit,
                client_max_tier=client_tier,
                **kwargs,
            )
        return await asyncio.to_thread(
            self.search,
            claim=claim,
            tier=tier,
            context_docs=context_docs,
            limit=limit,
            **kwargs,
        )


def get_retrieval_adapter(
    store: Optional[DualIndexStore] = None,
    embedder: Optional[BaseEmbedder] = None,
    tier3_adapter: Optional[Tier3WebAdapter] = None,
    server_max_tier: Optional[int] = None,
) -> RetrievalAdapter:
    """Factory helper to instantiate a configured RetrievalAdapter."""
    return RetrievalAdapter(
        store=store,
        embedder=embedder,
        tier3_adapter=tier3_adapter,
        server_max_tier=server_max_tier,
    )
