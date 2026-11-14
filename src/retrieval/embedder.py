"""Dense embedding abstractions and implementations.

Implements CPU-based embedding generation (REQ-RET-03, ADR-0002).
Primary model: google/embeddinggemma-2 (768-dimensional multimodal/text vector space).
Supports asymmetric task instruction prefixes (encode_query vs encode_document).
"""

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence


class BaseEmbedder(ABC):
    """Abstract interface for dense text embedders."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate normalized vector embeddings for corpus document passages."""
        pass

    @abstractmethod
    def embed_queries(self, queries: Sequence[str]) -> List[List[float]]:
        """Generate normalized vector embeddings for search claims/queries."""
        pass

    def embed_query(self, query: str) -> List[float]:
        """Convenience method to generate embedding for a single query."""
        results = self.embed_queries([query])
        if not results:
            return [0.0] * self.dimension
        return results[0]

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Backward-compatible alias delegating to embed_documents."""
        return self.embed_documents(texts)

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality of the embedding model."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier name."""
        pass


class EmbeddingGemmaEmbedder(BaseEmbedder):
    """CPU-based dense embedder powered by google/embeddinggemma-2.

    Generates 768-dimensional normalized embeddings using the lightweight text
    backbone on CPU threads to prevent GPU VRAM exhaustion (ADR-0002).
    Applies asymmetric task instruction prefixes via encode_document and encode_query.
    Supports Matryoshka Representation Learning (MRL) dimension truncation.
    """

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-2",
        dimension: int = 768,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._model: Optional[Any] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self) -> Any:
        """Lazy load SentenceTransformer with vision/audio disabled on CPU."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_name,
                device="cpu",
                config_kwargs={"vision_config": None, "audio_config": None},
            )
        return self._model

    def _postprocess(self, embeddings: Any) -> List[List[float]]:
        results: List[List[float]] = []
        for emb in embeddings:
            vec = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            if self._dimension < len(vec):
                # Matryoshka dimension truncation with re-normalization
                vec = vec[: self._dimension]
                norm = math.sqrt(sum(x * x for x in vec))
                if norm > 0:
                    vec = [x / norm for x in vec]
            results.append(vec)
        return results

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        if hasattr(model, "encode_document"):
            embeddings = model.encode_document(
                list(texts),
                normalize_embeddings=True,
                truncate_dim=self._dimension,
                convert_to_numpy=True,
            )
        else:
            try:
                embeddings = model.encode(
                    list(texts),
                    prompt_name="Document",
                    normalize_embeddings=True,
                    truncate_dim=self._dimension,
                    convert_to_numpy=True,
                )
            except Exception:
                embeddings = model.encode(
                    [f"search_document: {t}" for t in texts],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
        return self._postprocess(embeddings)

    def embed_queries(self, queries: Sequence[str]) -> List[List[float]]:
        if not queries:
            return []
        model = self._get_model()
        if hasattr(model, "encode_query"):
            embeddings = model.encode_query(
                list(queries),
                normalize_embeddings=True,
                truncate_dim=self._dimension,
                convert_to_numpy=True,
            )
        else:
            try:
                embeddings = model.encode(
                    list(queries),
                    prompt_name="SearchQuery",
                    normalize_embeddings=True,
                    truncate_dim=self._dimension,
                    convert_to_numpy=True,
                )
            except Exception:
                embeddings = model.encode(
                    [f"search_query: {q}" for q in queries],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
        return self._postprocess(embeddings)


class FastEmbedEmbedder(BaseEmbedder):
    """Lightweight CPU dense embedder powered by FastEmbed (MiniLM / BGE)."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._model: Optional[Any] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        embeddings = list(model.embed(list(texts)))
        return [e.tolist() for e in embeddings]

    def embed_queries(self, queries: Sequence[str]) -> List[List[float]]:
        if not queries:
            return []
        model = self._get_model()
        if hasattr(model, "query_embed"):
            embeddings = list(model.query_embed(list(queries)))
        else:
            embeddings = list(model.embed(list(queries)))
        return [e.tolist() for e in embeddings]


class MockEmbedder(BaseEmbedder):
    """Deterministic, zero-latency pseudo-embedder for offline testing."""

    def __init__(
        self,
        dimension: int = 768,
        model_name: str = "mock-embeddinggemma-2",
    ) -> None:
        self._dimension = dimension
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _compute_vector(self, text: str, prefix: str = "") -> List[float]:
        seed_hash = hashlib.sha256(f"{prefix}{text}".encode("utf-8")).digest()
        raw_vec = []
        for i in range(self._dimension):
            byte_val = seed_hash[i % len(seed_hash)]
            val = ((byte_val + i) % 100) / 50.0 - 1.0  # Range [-1.0, 1.0]
            raw_vec.append(val)

        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in raw_vec))
        if norm > 0:
            return [x / norm for x in raw_vec]
        return raw_vec

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._compute_vector(t, prefix="doc:") for t in texts]

    def embed_queries(self, queries: Sequence[str]) -> List[List[float]]:
        return [self._compute_vector(q, prefix="query:") for q in queries]


def get_default_embedder(
    model_name: str = "google/embeddinggemma-2",
    dimension: int = 768,
) -> BaseEmbedder:
    """Factory creating the configured dense embedder."""
    if "embeddinggemma" in model_name.lower():
        return EmbeddingGemmaEmbedder(model_name=model_name, dimension=dimension)
    elif "minilm" in model_name.lower() or "fastembed" in model_name.lower():
        return FastEmbedEmbedder(model_name=model_name, dimension=dimension)
    else:
        return EmbeddingGemmaEmbedder(model_name=model_name, dimension=dimension)
