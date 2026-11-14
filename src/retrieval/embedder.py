"""Dense embedding abstractions and implementations.

Implements CPU-based embedding generation (REQ-RET-03, ADR-0002).
Primary model: google/embeddinggemma-2 (768-dimensional multimodal vector space).
"""

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseEmbedder(ABC):
    """Abstract interface for dense text embedders."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized vector embeddings for a list of text strings."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality of the embedding model."""
        pass


class EmbeddingGemmaEmbedder(BaseEmbedder):
    """CPU-based dense embedder powered by google/embeddinggemma-2.

    Generates 768-dimensional normalized embeddings using the lightweight text
    backbone on CPU threads to prevent GPU VRAM exhaustion (ADR-0002).
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

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        # SentenceTransformer generates normalized unit vectors
        embeddings = model.encode(texts, normalize_embeddings=True)

        results: List[List[float]] = []
        for emb in embeddings:
            vec = emb.tolist()
            if self._dimension < len(vec):
                # Matryoshka dimension truncation with re-normalization
                vec = vec[: self._dimension]
                norm = math.sqrt(sum(x * x for x in vec))
                if norm > 0:
                    vec = [x / norm for x in vec]
            results.append(vec)
        return results

    @property
    def dimension(self) -> int:
        return self._dimension


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

    def _get_model(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        embeddings = list(model.embed(texts))
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension


class MockEmbedder(BaseEmbedder):
    """Deterministic, zero-latency pseudo-embedder for offline testing."""

    def __init__(self, dimension: int = 768) -> None:
        self._dimension = dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for text in texts:
            seed_hash = hashlib.sha256(text.encode("utf-8")).digest()
            raw_vec = []
            for i in range(self._dimension):
                byte_val = seed_hash[i % len(seed_hash)]
                val = ((byte_val + i) % 100) / 50.0 - 1.0  # Range [-1.0, 1.0]
                raw_vec.append(val)

            # L2 Normalize
            norm = math.sqrt(sum(x * x for x in raw_vec))
            if norm > 0:
                vec = [x / norm for x in raw_vec]
            else:
                vec = raw_vec
            results.append(vec)
        return results

    @property
    def dimension(self) -> int:
        return self._dimension


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
