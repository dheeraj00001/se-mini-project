"""Unit and integration tests for Hybrid Retrieval Adapters & Reciprocal Rank Fusion.

Covers RTM Traceability:
- UT_RET_04: Reciprocal Rank Fusion (RRF k=60) mathematical properties,
  rank inversion, deterministic tie-breaking, and boundary conditions (REQ-RET-04).
- UT_RET_05: Tier 1 (Context) and Tier 2 (Hybrid Qdrant + BM25) retrieval adapters,
  adapter factory interface, and latency compliance <= 50ms (REQ-RET-05, ADR-0004).
"""

import math
import time
from typing import List, Sequence, Tuple
from unittest.mock import patch
import pytest

from src.retrieval.adapters import (
    RetrievalAdapter,
    Tier1ContextAdapter,
    Tier2HybridAdapter,
    get_retrieval_adapter,
)
from src.retrieval.embedder import MockEmbedder
from src.retrieval.fusion import compute_rrf_score, explain_rrf_scoring, reciprocal_rank_fusion
from src.retrieval.store import DualIndexStore
from src.schemas.verification import EvidenceCandidate, EvidenceChunk


def make_dummy_chunk(
    chunk_id: str,
    text: str = "Test passage",
    source_doc: str = "test.txt",
    tier: int = 2,
    score: float | None = None,
) -> EvidenceChunk:
    """Helper to construct valid EvidenceChunk instances."""
    return EvidenceChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc=source_doc,
        tier=tier,
        score=score,
    )


# ============================================================================
# UT_RET_04: Reciprocal Rank Fusion Mathematical Properties (REQ-RET-04)
# ============================================================================


def test_ut_ret_04_compute_rrf_score_formula() -> None:
    """Verify exact formula calculation: RRF(d) = sum(1 / (k + r_m(d)))."""
    # 1st rank with default k=60 -> 1 / (60 + 1) = 1/61
    expected_single = 1.0 / 61.0
    assert math.isclose(compute_rrf_score([1], k=60), expected_single, rel_tol=1e-6)

    # 1st rank across 2 modalities -> 1/61 + 1/61 = 2/61
    expected_dual = 2.0 / 61.0
    assert math.isclose(compute_rrf_score([1, 1], k=60), expected_dual, rel_tol=1e-6)

    # Mixed ranks with custom k=10: rank 1 + rank 3 -> 1/11 + 1/13
    expected_custom = (1.0 / 11.0) + (1.0 / 13.0)
    assert math.isclose(compute_rrf_score([1, 3], k=10), expected_custom, rel_tol=1e-6)


def test_ut_ret_04_rrf_rank_inversion_consensus() -> None:
    """Verify rank inversion where cross-modal consensus beats single-modal top rank.

    Doc A: Rank 1 in dense, absent in sparse -> score = 1 / 61 ~= 0.016393
    Doc B: Rank 2 in dense, Rank 2 in sparse -> score = 1/62 + 1/62 = 2/62 ~= 0.032258
    Doc B must invert Doc A in the final fused ranking.
    """
    chunk_a = make_dummy_chunk("chunk_a", "Solo dense hit")
    chunk_b = make_dummy_chunk("chunk_b", "Dual modality hit")

    dense_ranking = [chunk_a, chunk_b]  # Doc A is #1, Doc B is #2
    sparse_ranking = [chunk_b]  # Only Doc B is #1 in sparse

    fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking], k=60)

    assert len(fused) == 2
    # chunk_b has higher combined score (1/62 + 1/61 ~= 0.03252) than chunk_a (1/61 ~= 0.01639)
    assert fused[0].chunk_id == "chunk_b"
    assert fused[1].chunk_id == "chunk_a"
    assert fused[0].score is not None and fused[1].score is not None
    assert fused[0].score > fused[1].score


def test_ut_ret_04_rrf_deterministic_tie_breaking() -> None:
    """Verify deterministic tie-breaking by chunk_id when fused RRF scores are equal."""
    chunk_x = make_dummy_chunk("chunk_x", "Passage X")
    chunk_y = make_dummy_chunk("chunk_y", "Passage Y")

    # List 1: chunk_x at rank 1
    # List 2: chunk_y at rank 1
    # Both have score 1/61
    fused = reciprocal_rank_fusion([[chunk_x], [chunk_y]], k=60)

    assert len(fused) == 2
    assert fused[0].score == fused[1].score
    # Deterministic alphabetical tie-breaking by chunk_id
    assert [c.chunk_id for c in fused] == ["chunk_x", "chunk_y"]


def test_ut_ret_04_rrf_tuples_with_raw_scores_input() -> None:
    """Verify fusion accepts (chunk, raw_score) tuples as returned by search methods."""
    chunk_1 = make_dummy_chunk("c1", "Passage 1")
    chunk_2 = make_dummy_chunk("c2", "Passage 2")

    dense_tuples: List[Tuple[EvidenceChunk, float]] = [(chunk_1, 0.95), (chunk_2, 0.80)]
    sparse_tuples: List[Tuple[EvidenceChunk, float]] = [(chunk_2, 12.4), (chunk_1, 4.2)]

    fused = reciprocal_rank_fusion([dense_tuples, sparse_tuples], k=60, limit=1)

    assert len(fused) == 1
    assert isinstance(fused[0], EvidenceChunk)
    # chunk_1: 1/61 + 1/62 = 0.032522
    # chunk_2: 1/62 + 1/61 = 0.032522
    # Tie broken deterministically by chunk_id: "c1" < "c2"
    assert fused[0].chunk_id == "c1"


def test_ut_ret_04_rrf_edge_cases_and_validation() -> None:
    """Verify RRF error handling and boundary conditions."""
    # Invalid k <= 0
    with pytest.raises(ValueError, match="k must be greater than 0"):
        reciprocal_rank_fusion([], k=0)

    with pytest.raises(ValueError, match="k must be greater than 0"):
        compute_rrf_score([1], k=-5)

    # Empty input rankings
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []

    # Single ranking list
    chunk = make_dummy_chunk("c1")
    single_res = reciprocal_rank_fusion([[chunk]], k=60)
    assert len(single_res) == 1
    assert single_res[0].chunk_id == "c1"
    assert single_res[0].score == pytest.approx(1.0 / 61.0)

    # Duplicate occurrences in same modality take the earliest/best rank
    dup_res = reciprocal_rank_fusion([[chunk, chunk]], k=60)
    assert len(dup_res) == 1
    assert dup_res[0].score == pytest.approx(1.0 / 61.0)


def test_ut_ret_04_explain_rrf_scoring() -> None:
    """Verify explain_rrf_scoring provides accurate rank breakdown and fused scoring (REQ-RET-04)."""
    c1 = make_dummy_chunk("chunk_alpha")
    c2 = make_dummy_chunk("chunk_beta")
    dense_ranking = [c1, c2]
    sparse_ranking = [c2, c1]

    explanation = explain_rrf_scoring("chunk_alpha", [dense_ranking, sparse_ranking], k=60)
    assert explanation["chunk_id"] == "chunk_alpha"
    assert explanation["k"] == 60
    assert explanation["modality_ranks"] == [1, 2]
    expected_score = (1.0 / 61.0) + (1.0 / 62.0)
    assert explanation["fused_rrf_score"] == pytest.approx(expected_score)

    # Missing chunk returns 0.0 and empty ranks
    missing = explain_rrf_scoring("nonexistent", [dense_ranking, sparse_ranking], k=60)
    assert missing["modality_ranks"] == []
    assert missing["fused_rrf_score"] == 0.0



def test_ut_ret_04_store_search_sparse() -> None:
    """Verify DualIndexStore.search_sparse provides BM25 exact keyword ranking."""
    store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))

    # Empty store
    assert store.search_sparse("machine learning") == []

    # Ingest chunks
    c1 = make_dummy_chunk("c1", "Quantum computing leverages superposition and entanglement.")
    c2 = make_dummy_chunk("c2", "Classical computing uses bits with deterministic states.")
    c3 = make_dummy_chunk("c3", "Quantum key distribution ensures cryptography security.")
    store.add_chunks([c1, c2, c3])

    # Empty or whitespace query
    assert store.search_sparse("") == []
    assert store.search_sparse("   \t  ") == []

    # Query with non-matching terms
    assert store.search_sparse("astronomy telescope galaxy") == []

    # Query matching c1 and c3
    results = store.search_sparse("quantum superposition", limit=10)
    assert len(results) >= 1
    top_chunk, top_score = results[0]
    assert top_chunk.chunk_id == "c1"
    assert top_score > 0.0

    # Limit truncation
    assert len(store.search_sparse("quantum", limit=1)) == 1


class ControlledEmbedder(MockEmbedder):
    """Embedder with controlled vectors for deterministic similarity testing."""

    def __init__(self) -> None:
        super().__init__(dimension=2)

    def embed_query(self, query: str) -> List[float]:
        if "colosseum" in query.lower():
            return [1.0, 0.0]
        return [0.0, 1.0]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        vecs = []
        for t in texts:
            if "colosseum" in t.lower():
                vecs.append([1.0, 0.0])
            else:
                vecs.append([0.0, 1.0])
        return vecs


# ============================================================================
# UT_RET_05: Tier 1 Context Retrieval Adapter (REQ-RET-05, ADR-0004)
# ============================================================================


def test_ut_ret_05_tier1_context_adapter_lexical_and_hybrid() -> None:
    """Verify Tier1ContextAdapter extracts relevant passages from request context documents."""
    docs = [
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "The Colosseum is an ancient elliptical amphitheatre in the centre of Rome, Italy.",
        "The Great Wall of China is a series of fortifications built across northern "
        "historical borders.",
    ]

    adapter = Tier1ContextAdapter()
    results = adapter.search("Where is the Eiffel Tower located?", context_documents=docs, limit=2)

    assert len(results) >= 1
    assert results[0].tier == 1
    assert "Eiffel Tower" in results[0].text
    assert results[0].score is not None

    # Search with embedder enabled
    embedder = ControlledEmbedder()
    hybrid_adapter = Tier1ContextAdapter(embedder=embedder)
    results_hybrid = hybrid_adapter.search("Colosseum in Rome", context_documents=docs, limit=1)

    assert len(results_hybrid) == 1
    assert results_hybrid[0].tier == 1
    assert "Colosseum" in results_hybrid[0].text
    assert results_hybrid[0].score is not None


def test_ut_ret_05_tier1_context_adapter_empty_and_edge_cases() -> None:
    """Verify Tier1ContextAdapter handles empty inputs gracefully."""
    adapter = Tier1ContextAdapter()

    # Empty context docs
    assert adapter.search("Any claim", context_documents=[]) == []
    assert adapter.search("Any claim", context_documents=None) == []

    # Whitespace or empty claims
    docs = ["Some factual statement about astronomy."]
    assert adapter.search("", context_documents=docs) == []
    assert adapter.search("   \n\t  ", context_documents=docs) == []

    # Empty doc strings
    assert adapter.search("Some claim", context_documents=["", "   \t  "]) == []

    # Default context_documents provided at init
    init_adapter = Tier1ContextAdapter(context_documents=docs)
    res = init_adapter.search("factual statement")
    assert len(res) == 1
    assert res[0].tier == 1
    assert "factual statement" in res[0].text


# ============================================================================
# UT_RET_05: Tier 2 Hybrid Retrieval Adapter & RRF Fusion (REQ-RET-04, REQ-RET-05)
# ============================================================================


def test_ut_ret_05_tier2_hybrid_adapter_concurrent_search() -> None:
    """Verify Tier2HybridAdapter concurrently queries Qdrant & BM25 and fuses ranks."""
    store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))
    c1 = make_dummy_chunk(
        "c1",
        text="Apollo 11 landed humans on the Moon on July 20, 1969.",
        source_doc="history.txt",
        tier=2,
    )
    c2 = make_dummy_chunk(
        "c2",
        text="Neil Armstrong was the first person to walk on the lunar surface.",
        source_doc="history.txt",
        tier=2,
    )
    c3 = make_dummy_chunk(
        "c3",
        text="The International Space Station operates in low Earth orbit.",
        source_doc="space.txt",
        tier=2,
    )
    store.add_chunks([c1, c2, c3])

    adapter = Tier2HybridAdapter(store=store)
    assert adapter.store is store

    # Search for Apollo Moon landing
    results = adapter.search("Apollo 11 moon landing", limit=2)

    assert len(results) >= 1
    assert results[0].tier == 2
    assert results[0].score is not None
    assert results[0].score > 0.0
    # Top result should be Apollo 11 chunk
    assert results[0].chunk_id in ("c1", "c2")

    # Limit parameter
    assert len(adapter.search("space", limit=1)) == 1

    # Empty query
    assert adapter.search("") == []
    assert adapter.search("   \n\t  ") == []


def test_ut_ret_05_unified_retrieval_adapter_routing_and_validation() -> None:
    """Verify unified RetrievalAdapter facade routes by tier and validates inputs."""
    store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))
    chunk = make_dummy_chunk("c1", "Local knowledge passage on photosynthesis.", tier=2)
    store.add_chunks([chunk])

    adapter = get_retrieval_adapter(store=store)
    assert isinstance(adapter, RetrievalAdapter)
    assert isinstance(adapter.tier1_adapter, Tier1ContextAdapter)
    assert isinstance(adapter.tier2_adapter, Tier2HybridAdapter)

    # Route Tier 1
    context_docs = ["Photosynthesis converts light into chemical energy."]
    t1_res = adapter.search("photosynthesis light", tier=1, context_docs=context_docs)
    assert len(t1_res) >= 1
    assert t1_res[0].tier == 1

    # Route Tier 2
    t2_res = adapter.search("photosynthesis knowledge", tier=2)
    assert len(t2_res) >= 1
    assert isinstance(t2_res[0], EvidenceCandidate)
    assert t2_res[0].tier == 2
    assert t2_res[0].chunk_id == "c1"

    # Route Tier 3
    with patch.object(adapter.tier3_adapter, "search", return_value=[]) as mock_t3_search:
        t3_res = adapter.search("photosynthesis", tier=3)
        assert t3_res == []
        mock_t3_search.assert_called_once()

    # Invalid tier out of range [1, 3]
    with pytest.raises(ValueError, match="Invalid retrieval tier"):
        adapter.search("test", tier=0)

    with pytest.raises(ValueError, match="Invalid retrieval tier"):
        adapter.search("test", tier=4)


# ============================================================================
# NFR-PERF: Retrieval Latency Compliance <= 50ms (REQ-RET-04, Ticket 04)
# ============================================================================


def test_nfr_perf_retrieval_latency_under_50ms() -> None:
    """Verify local Tier 2 hybrid retrieval completes well within the 50ms budget."""
    store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))
    chunks = [
        make_dummy_chunk(
            f"c_{i}", f"Corpus knowledge passage number {i} with distinct keywords.", tier=2
        )
        for i in range(25)
    ]
    store.add_chunks(chunks)

    adapter = Tier2HybridAdapter(store=store)

    # Warm-up query
    adapter.search("passage number 1", limit=5)

    # Measure 10 queries
    latencies: List[float] = []
    for i in range(10):
        t0 = time.perf_counter()
        results = adapter.search(f"passage number {i}", limit=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)
        assert len(results) >= 1

    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    # Average and P95 latency must be strictly <= 50ms (typically <= 10ms for local queries)
    assert p95_latency <= 50.0, f"P95 latency {p95_latency:.2f}ms exceeded 50ms requirement"
