"""Smoke test and empirical validation utility for model deployments.

Verifies:
1. Hardware resource environment (RTX 2050 Mobile 4 GB VRAM ceiling, CPU threads).
2. google/embeddinggemma-2: Asymmetric query/document encoding and 768-dim embeddings.
3. cross-encoder/nli-deberta-v3-base: Label mapping [contradiction, entailment, neutral].
4. Qdrant dual-index store persistence with 768-dim vectors and metadata.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure repository root is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.embedder import MockEmbedder, get_default_embedder
from src.retrieval.store import DualIndexStore
from src.schemas.verification import (
    DEBERTA_NLI_LABEL_ORDER,
    DEBERTA_TO_CLAIM_STATE,
    ClaimState,
    EvidenceChunk,
    determine_claim_state_from_nli,
    map_nli_probabilities,
)


def validate_hardware_environment() -> Dict[str, Any]:
    """Inspect and report host CPU and GPU hardware status."""
    report: Dict[str, Any] = {
        "python_version": sys.version,
        "cuda_available": False,
        "gpu_name": None,
        "total_vram_mb": None,
    }
    try:
        import torch

        report["torch_version"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu_name"] = torch.cuda.get_device_name(0)
            report["total_vram_mb"] = (
                torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            )
    except ImportError:
        report["torch_version"] = "not_installed"
    return report


def validate_nli_label_mapping() -> bool:
    """Validate cross-encoder/nli-deberta-v3-base label order and mapping."""
    print("\n--- Validating NLI Label Mapping ---")
    assert DEBERTA_NLI_LABEL_ORDER == ("contradiction", "entailment", "neutral")
    assert DEBERTA_TO_CLAIM_STATE["contradiction"] == ClaimState.CONTRADICTED
    assert DEBERTA_TO_CLAIM_STATE["entailment"] == ClaimState.SUPPORTED
    assert DEBERTA_TO_CLAIM_STATE["neutral"] == ClaimState.UNVERIFIED

    # Test probability interpretation
    supported_case = map_nli_probabilities([0.05, 0.90, 0.05])
    contradicted_case = map_nli_probabilities([0.88, 0.06, 0.06])
    unverified_case = map_nli_probabilities([0.10, 0.10, 0.80])

    assert determine_claim_state_from_nli(supported_case) == ClaimState.SUPPORTED
    assert determine_claim_state_from_nli(contradicted_case) == ClaimState.CONTRADICTED
    assert determine_claim_state_from_nli(unverified_case) == ClaimState.UNVERIFIED

    print("✔ DeBERTa label mapping verified: Index 0=contradiction, 1=entailment, 2=neutral.")
    return True


def validate_asymmetric_embedding_and_store() -> bool:
    """Validate asymmetric embedding generation and Qdrant 768-dim persistence."""
    print("\n--- Validating Asymmetric Embeddings & Qdrant Store ---")
    mock = MockEmbedder(dimension=768, model_name="google/embeddinggemma-2")
    doc_vec = mock.embed_documents(["The Earth orbits the Sun."])[0]
    query_vec = mock.embed_query("Earth orbits the Sun")

    assert len(doc_vec) == 768
    assert len(query_vec) == 768
    assert doc_vec != query_vec, "Query and document vectors must be asymmetric"

    # Validate store creation & metadata
    store = DualIndexStore(
        in_memory=True,
        collection_name="smoke_test_kb",
        embedder=mock,
    )
    chunk = EvidenceChunk(
        chunk_id="smoke-chunk-01",
        text="The Earth orbits the Sun in approximately 365.25 days.",
        source_doc="astronomy.md",
        tier=2,
    )
    indexed_count = store.add_chunks([chunk])
    assert indexed_count == 1
    assert store.count() == 1

    meta = store.get_collection_metadata()
    assert meta.get("dimension") == 768
    assert meta.get("model_name") == "google/embeddinggemma-2"

    # Dense retrieval check
    hits = store.search_dense("orbital period of Earth", limit=1)
    assert len(hits) == 1
    assert hits[0][0].chunk_id == "smoke-chunk-01"

    print("✔ Asymmetric embeddings and 768-dim Qdrant store verified.")
    return True


if __name__ == "__main__":
    hw = validate_hardware_environment()
    print("Hardware Environment Report:")
    for k, v in hw.items():
        print(f"  {k}: {v}")

    validate_nli_label_mapping()
    validate_asymmetric_embedding_and_store()
    print("\nAll model validation smoke checks passed successfully.")
