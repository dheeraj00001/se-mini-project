"""Dense embedding abstractions and implementations.

Implements CPU-based embedding generation (REQ-RET-03, ADR-0002).
"""

import hashlib
import math
from abc import ABC, abstractmethod
from typing import List


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


class FastEmbedEmbedder(BaseEmbedder):
    """CPU-based dense embedder powered by FastEmbed (MiniLM / BGE)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=self._model_name)
        self._dimension = 384  # all-MiniLM-L6-v2 vector dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = list(self._model.embed(texts))
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension


class MockEmbedder(BaseEmbedder):
    """Deterministic, zero-latency pseudo-embedder for offline testing."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for text in texts:
            # Generate deterministic floats from text hash
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
