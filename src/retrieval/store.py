"""Synchronized dual-index persistence for Qdrant vector store and BM25 index.

Implements REQ-RET-03, ADR-0002, and ADR-0007:
- Local file-backed Qdrant vector database (default: ./data/qdrant_storage).
- Disk-serialized BM25 index via joblib (default: ./data/bm25_index.joblib).
- Shared deterministic SHA-256 chunk IDs.
- Duplicate document skipping and persistence across restarts.
- Automatic vector dimension migration (384 -> 768) and metadata persistence.
- Asymmetric embedding pipeline (embed_documents for corpus, embed_query for search).
"""

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from src.retrieval.embedder import BaseEmbedder, get_default_embedder
from src.schemas.verification import EvidenceChunk


class DualIndexStore:
    """Manages synchronized persistence across Qdrant vector store and BM25 index."""

    def __init__(
        self,
        qdrant_path: Optional[str] = "./data/qdrant_storage",
        bm25_path: Optional[str] = "./data/bm25_index.joblib",
        collection_name: str = "knowledge_base",
        embedder: Optional[BaseEmbedder] = None,
        in_memory: bool = False,
    ) -> None:
        self._in_memory = in_memory
        self._collection_name = collection_name
        self._bm25_path = Path(bm25_path) if bm25_path else None
        self._embedder: BaseEmbedder = embedder if embedder is not None else get_default_embedder()

        # In-memory indices & metadata cache
        self._chunks: Dict[str, EvidenceChunk] = {}
        self._doc_hashes: Dict[str, str] = {}
        self._bm25: Optional[BM25Okapi] = None

        # Load BM25 and chunk metadata if present on disk prior to vector sync
        if not self._in_memory and self._bm25_path and self._bm25_path.is_file():
            self._load_bm25()

        # Initialize Qdrant Client
        if self._in_memory or not qdrant_path:
            self._client = QdrantClient(":memory:")
        else:
            q_dir = Path(qdrant_path)
            q_dir.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(q_dir))

        self._ensure_qdrant_collection()

    @property
    def embedder(self) -> BaseEmbedder:
        """Return the active dense embedder."""
        return self._embedder

    @property
    def collection_name(self) -> str:
        """Return the Qdrant collection name."""
        return self._collection_name

    def get_collection_metadata(self) -> Dict[str, Any]:
        """Retrieve stored collection metadata from Qdrant."""
        try:
            info = self._client.get_collection(collection_name=self._collection_name)
            if hasattr(info, "config") and hasattr(info.config, "metadata"):
                meta = info.config.metadata
                if isinstance(meta, dict):
                    return meta
        except Exception:
            pass
        return {
            "model_name": self._embedder.model_name,
            "dimension": self._embedder.dimension,
        }

    def _ensure_qdrant_collection(self) -> None:
        """Create the Qdrant collection or recreate it if dimension differs."""
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection_name in collections:
            info = self._client.get_collection(collection_name=self._collection_name)
            existing_size: Optional[int] = None
            if hasattr(info, "config") and hasattr(info.config, "params"):
                params = info.config.params
                if hasattr(params, "vectors") and params.vectors:
                    if hasattr(params.vectors, "size"):
                        existing_size = params.vectors.size
                    elif isinstance(params.vectors, dict) and "" in params.vectors:
                        existing_size = params.vectors[""].size

            if existing_size is not None and existing_size != self._embedder.dimension:
                # Dimension mismatch (e.g., 384 from MiniLM vs 768 from EmbeddingGemma-2)
                # On Windows local file Qdrant, close open SQLite handles before deletion
                try:
                    if hasattr(self._client, "_client") and hasattr(
                        self._client._client, "collections"
                    ):
                        coll = self._client._client.collections.get(self._collection_name)
                        if coll is not None and hasattr(coll, "close"):
                            coll.close()
                except Exception:
                    pass

                self._client.delete_collection(collection_name=self._collection_name)
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=self._embedder.dimension,
                        distance=Distance.COSINE,
                    ),
                    metadata={
                        "model_name": self._embedder.model_name,
                        "dimension": self._embedder.dimension,
                    },
                )
                self._reembed_all_chunks()
                return

        if self._collection_name not in collections:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._embedder.dimension,
                    distance=Distance.COSINE,
                ),
                metadata={
                    "model_name": self._embedder.model_name,
                    "dimension": self._embedder.dimension,
                },
            )
            if self._chunks:
                self._reembed_all_chunks()

    def _reembed_all_chunks(self) -> None:
        """Re-embed all cached chunks with current embedder and upsert to Qdrant."""
        if not self._chunks:
            return
        chunk_list = list(self._chunks.values())
        texts = [c.text for c in chunk_list]
        vectors = self._embedder.embed_documents(texts)

        points = []
        for chunk, vec in zip(chunk_list, vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))
            payload = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source_doc": chunk.source_doc,
                "tier": chunk.tier,
                "metadata": chunk.metadata,
            }
            points.append(PointStruct(id=point_id, vector=vec, payload=payload))

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

    def _load_bm25(self) -> None:
        """Load serialized BM25 index and chunk state from disk."""
        if not self._bm25_path or not self._bm25_path.is_file():
            return
        state = joblib.load(self._bm25_path)
        self._chunks = state.get("chunks", {})
        self._doc_hashes = state.get("doc_hashes", {})
        self._bm25 = state.get("bm25")

        if self._bm25 is None and self._chunks:
            corpus = [c.text.lower().split() for c in self._chunks.values()]
            self._bm25 = BM25Okapi(corpus)

    def _save_bm25(self) -> None:
        """Serialize BM25 index and chunk state to disk."""
        if self._in_memory or not self._bm25_path:
            return
        self._bm25_path.parent.mkdir(parents=True, exist_ok=True)
        state: Dict[str, Any] = {
            "chunks": self._chunks,
            "doc_hashes": self._doc_hashes,
            "bm25": self._bm25,
        }
        joblib.dump(state, self._bm25_path)

    def has_document(self, source_doc: str, doc_hash: str) -> bool:
        """Check if a document has already been indexed with the given hash."""
        return self._doc_hashes.get(source_doc) == doc_hash

    def add_chunks(
        self,
        chunks: List[EvidenceChunk],
        doc_hash: Optional[str] = None,
    ) -> int:
        """Synchronously index chunks in both Qdrant and BM25.

        Args:
            chunks: List of EvidenceChunk instances to index.
            doc_hash: Optional document SHA-256 hash for deduplication.

        Returns:
            Number of newly indexed chunks.
        """
        if not chunks:
            return 0

        source_doc = chunks[0].source_doc

        # Skip if identical document has already been ingested
        if doc_hash and self.has_document(source_doc, doc_hash):
            return 0

        # Filter out chunks that are already indexed
        new_chunks = [c for c in chunks if c.chunk_id not in self._chunks]
        if not new_chunks:
            if doc_hash:
                self._doc_hashes[source_doc] = doc_hash
                self._save_bm25()
            return 0

        # 1. Dense Embedding Generation (using asymmetric document embedding)
        texts = [c.text for c in new_chunks]
        vectors = self._embedder.embed_documents(texts)

        points = []
        for chunk, vec in zip(new_chunks, vectors):
            # Deterministic UUID mapping for Qdrant compatibility
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))
            payload = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source_doc": chunk.source_doc,
                "tier": chunk.tier,
                "metadata": chunk.metadata,
            }
            points.append(PointStruct(id=point_id, vector=vec, payload=payload))

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

        # 2. Update chunk cache & BM25 sparse index
        for c in new_chunks:
            self._chunks[c.chunk_id] = c

        if doc_hash:
            self._doc_hashes[source_doc] = doc_hash

        # Rebuild BM25 with full corpus
        corpus = [c.text.lower().split() for c in self._chunks.values()]
        self._bm25 = BM25Okapi(corpus)

        # 3. Persist state to disk
        self._save_bm25()

        return len(new_chunks)

    def search_dense(self, query: str, limit: int = 10) -> List[Tuple[EvidenceChunk, float]]:
        """Search Qdrant using asymmetric query embedding.

        Returns a list of (EvidenceChunk, cosine_similarity_score) pairs.
        """
        if not self._chunks:
            return []

        query_vec = self._embedder.embed_query(query)
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vec,
            limit=limit,
        )

        results: List[Tuple[EvidenceChunk, float]] = []
        for point in response.points:
            chunk_id = point.payload.get("chunk_id") if point.payload else None
            if chunk_id and chunk_id in self._chunks:
                chunk = self._chunks[chunk_id]
                score = float(point.score) if point.score is not None else 0.0
                results.append((chunk, score))

        return results

    def get_chunk(self, chunk_id: str) -> Optional[EvidenceChunk]:
        """Retrieve an EvidenceChunk by its SHA-256 chunk ID."""
        return self._chunks.get(chunk_id)

    def get_all_chunks(self) -> List[EvidenceChunk]:
        """Return all indexed EvidenceChunk instances."""
        return list(self._chunks.values())

    def count(self) -> int:
        """Return total number of unique chunks indexed."""
        return len(self._chunks)

    def close(self) -> None:
        """Close Qdrant client connection."""
        try:
            self._client.close()
        except Exception:
            pass
