"""Reciprocal Rank Fusion (RRF) for multi-modal retrieval rank aggregation.

Implements REQ-RET-04 and ADR-0007:
Combines dense vector search ranks from Qdrant with sparse keyword ranks from BM25:
    RRF(d) = sum_{m in M} (1 / (k + r_m(d)))
where default k = 60 and r_m(d) is the 1-based rank in modality m.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

from src.schemas.verification import EvidenceChunk

DEFAULT_RRF_K: int = 60


def compute_rrf_score(ranks: Sequence[int], k: int = DEFAULT_RRF_K) -> float:
    """Calculate the Reciprocal Rank Fusion score across multiple modality ranks.

    Args:
        ranks: Sequence of 1-based ranks (e.g. [1, 2]). Rank <= 0 or missing are ignored.
        k: Smoothing constant (standard: 60). Must be positive.

    Returns:
        Aggregated RRF score (float).

    Raises:
        ValueError: If k <= 0.
    """
    if k <= 0:
        raise ValueError(f"k must be greater than 0, got {k}")
    return sum(1.0 / (k + r) for r in ranks if r > 0)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Union[EvidenceChunk, Tuple[EvidenceChunk, float]]]],
    k: int = DEFAULT_RRF_K,
    limit: Optional[int] = None,
) -> List[EvidenceChunk]:
    """Merge and rerank multiple candidate rankings using Reciprocal Rank Fusion.

    Args:
        rankings: Sequence of ranked candidate lists from distinct modalities
            (e.g. [dense_results, sparse_results]). Each list can contain either
            EvidenceChunk objects or (EvidenceChunk, raw_score) tuples, ordered
            from best (rank 1) to worst.
        k: RRF constant (default: 60).
        limit: Optional maximum number of fused candidates to return.

    Returns:
        Sorted list of EvidenceChunk instances with their `.score` field set
        to the combined RRF score. Ties are broken deterministically by chunk_id.

    Raises:
        ValueError: If k <= 0.
    """
    if k <= 0:
        raise ValueError(f"k must be greater than 0, got {k}")

    if not rankings:
        return []

    # Map chunk_id to (EvidenceChunk, list of 1-based ranks)
    chunk_map: Dict[str, EvidenceChunk] = {}
    chunk_ranks: Dict[str, List[int]] = {}

    for ranked_list in rankings:
        seen_in_modality = set()
        for idx, item in enumerate(ranked_list):
            if isinstance(item, tuple):
                chunk, _ = item
            else:
                chunk = item

            if chunk.chunk_id in seen_in_modality:
                # Modality has duplicate entry: earliest (best) rank is preserved
                continue
            seen_in_modality.add(chunk.chunk_id)

            rank = idx + 1  # 1-based rank
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk
                chunk_ranks[chunk.chunk_id] = []
            chunk_ranks[chunk.chunk_id].append(rank)

    if not chunk_map:
        return []

    # Calculate fused scores
    scored_candidates: List[Tuple[float, str, EvidenceChunk]] = []
    for chunk_id, chunk in chunk_map.items():
        ranks = chunk_ranks[chunk_id]
        score = compute_rrf_score(ranks, k=k)
        scored_candidates.append((score, chunk_id, chunk))

    # Sort descending by score (-score), then ascending by chunk_id for deterministic tie breaking
    scored_candidates.sort(key=lambda item: (-item[0], item[1]))

    if limit is not None and limit >= 0:
        scored_candidates = scored_candidates[:limit]

    # Return new EvidenceChunk instances with updated score
    results: List[EvidenceChunk] = []
    for score, chunk_id, chunk in scored_candidates:
        updated_chunk = chunk.model_copy(update={"score": score})
        results.append(updated_chunk)

    return results
