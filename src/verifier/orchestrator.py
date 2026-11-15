"""NLI-Aware Waterfall Verification Orchestrator and Pipeline Integration.

Implements REQ-GW-02, REQ-RET-05, REQ-VER-05, REQ-VER-06, ADR-0004, ADR-0005.

Core Responsibilities:
1. Deterministic claim segmentation via ClaimSegmenter (REQ-VER-01..03).
2. Sequential Waterfall Evidence Retrieval Loop:
   - Tier 1: In-context documents (Tier1ContextAdapter). Short-circuits if decisive.
   - Tier 2: Local hybrid knowledge base (Qdrant + BM25 via Tier2HybridAdapter).
   - Tier 3: External web search (staged for Sprint 2).
   - Unverified fallback: when no decisive evidence is found (evidence_text = None).
3. 3-way Natural Language Inference verification on CPU (cross-encoder/nli-deberta-v3-base).
4. Document-level reliability rates & dual-flag safety gating (is_safe, needs_review).
5. Comprehensive schema-compliant VerificationResponse compilation.
"""

import logging
from typing import Any, List, Optional, Sequence, cast
from src.gateway.middleware import get_server_max_tier
from src.retrieval.adapters import (
    RetrievalAdapter,
    get_retrieval_adapter,
)
from src.retrieval.store import DualIndexStore
from src.schemas.verification import (
    ClaimReport,
    ClaimSegment,
    ClaimState,
    EvidenceChunk,
    MitigationMode,
    VerificationRequest,
    VerificationResponse,
)
from src.verifier.nli import (
    BaseNLIVerifier,
    compute_document_rates,
    determine_claim_state,
    evaluate_safety_gate,
    get_nli_verifier,
)
from src.verifier.segmenter import ClaimSegmenter

logger = logging.getLogger(__name__)


class WaterfallOrchestrator:
    """Orchestrates claim extraction, sequential waterfall retrieval, and NLI verification."""

    def __init__(
        self,
        retrieval_adapter: Optional[RetrievalAdapter] = None,
        nli_verifier: Optional[BaseNLIVerifier] = None,
        segmenter: Optional[ClaimSegmenter] = None,
        store: Optional[DualIndexStore] = None,
        entailment_threshold: float = 0.70,
        contradiction_threshold: float = 0.70,
        severe_contradiction_threshold: float = 0.85,
        server_max_tier: Optional[int] = None,
        top_k_per_tier: int = 3,
        rewriter: Optional[Any] = None,
        mitigation_pipeline: Optional[Any] = None,
        enable_mitigation: bool = True,
        raise_on_block: bool = True,
    ) -> None:
        self._server_max_tier = (
            server_max_tier if server_max_tier is not None else get_server_max_tier()
        )

        if store is not None:
            self._store = store
        elif (
            retrieval_adapter is not None
            and hasattr(retrieval_adapter, "tier2_adapter")
            and hasattr(retrieval_adapter.tier2_adapter, "store")
        ):
            self._store = retrieval_adapter.tier2_adapter.store
        else:
            self._store = DualIndexStore(in_memory=True)

        self._retrieval_adapter = (
            retrieval_adapter
            if retrieval_adapter is not None
            else get_retrieval_adapter(store=self._store, server_max_tier=self._server_max_tier)
        )
        self._nli_verifier = nli_verifier if nli_verifier is not None else get_nli_verifier()
        self._segmenter = segmenter if segmenter is not None else ClaimSegmenter()
        self._entailment_threshold = entailment_threshold
        self._contradiction_threshold = contradiction_threshold
        self._severe_contradiction_threshold = severe_contradiction_threshold
        self._top_k_per_tier = top_k_per_tier
        self._rewriter = rewriter
        self._enable_mitigation = enable_mitigation
        self._raise_on_block = raise_on_block

        if mitigation_pipeline is not None:
            self._mitigation_pipeline = mitigation_pipeline
        elif self._enable_mitigation:
            from src.mitigation.pipeline import MitigationPipeline
            from src.mitigation.rewriter import TargetedRewriter

            active_rewriter = self._rewriter if self._rewriter is not None else TargetedRewriter()
            self._mitigation_pipeline = MitigationPipeline(
                rewriter=active_rewriter,
                orchestrator=self,
                contradiction_threshold=self._contradiction_threshold,
                severe_contradiction_threshold=self._severe_contradiction_threshold,
            )
        else:
            self._mitigation_pipeline = None

    @property
    def retrieval_adapter(self) -> RetrievalAdapter:
        """Return the underlying RetrievalAdapter."""
        return self._retrieval_adapter

    @property
    def nli_verifier(self) -> BaseNLIVerifier:
        """Return the underlying NLI verifier."""
        return self._nli_verifier

    @property
    def segmenter(self) -> ClaimSegmenter:
        """Return the ClaimSegmenter instance."""
        return self._segmenter

    @property
    def store(self) -> DualIndexStore:
        """Return the underlying DualIndexStore."""
        return self._store

    @property
    def mitigation_pipeline(self) -> Optional[Any]:
        """Return the underlying MitigationPipeline instance."""
        return self._mitigation_pipeline

    @property
    def rewriter(self) -> Optional[Any]:
        """Return the underlying TargetedRewriter instance."""
        return self._rewriter

    @property
    def server_max_tier(self) -> int:
        """Return the server-side maximum retrieval tier ceiling."""
        return self._server_max_tier

    def _evaluate_chunks_for_claim(
        self,
        claim: ClaimSegment,
        chunks: Sequence[EvidenceChunk],
        tier: int,
    ) -> Optional[ClaimReport]:
        """Evaluate candidate evidence chunks against an atomic claim using NLI."""
        for chunk in chunks:
            probs_list = self._nli_verifier.predict_proba([(chunk.text, claim.claim_text)])
            probs = (
                probs_list[0]
                if probs_list
                else {"contradiction": 0.0, "entailment": 0.0, "neutral": 1.0}
            )
            state = determine_claim_state(
                probabilities=probs,
                premise_text=chunk.text,
                entailment_threshold=self._entailment_threshold,
                contradiction_threshold=self._contradiction_threshold,
            )
            if state in (ClaimState.SUPPORTED, ClaimState.CONTRADICTED):
                return ClaimReport(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    original_sentence=claim.original_sentence,
                    state=state,
                    probabilities=probs,
                    evidence_tier=tier,
                    evidence_text=chunk.text,
                )
        return None

    def verify(
        self, request: VerificationRequest, _is_secondary: bool = False
    ) -> VerificationResponse:
        """Execute end-to-end NLI-aware Waterfall verification on incoming text."""
        # 1. Deterministic claim segmentation (REQ-VER-01..03)
        claim_segments = self._segmenter.extract_claims(request.text)

        # Handle text with zero factual claims (e.g. greetings, filler, empty)
        if not claim_segments:
            rates = compute_document_rates([])
            return VerificationResponse(
                is_safe=True,
                needs_review=False,
                hallucination_rate=rates["hallucination_rate"],
                verification_rate=rates["verification_rate"],
                unverified_rate=rates["unverified_rate"],
                claims=[],
                sanitized_text=request.text,
            )

        # Execute Waterfall loop per claim
        claims_reports: List[ClaimReport] = []
        if request.max_retrieval_tier > self._server_max_tier:
            logger.warning(
                "Client requested retrieval tier %d exceeding server ceiling %d; capping at %d",
                request.max_retrieval_tier,
                self._server_max_tier,
                self._server_max_tier,
            )
            effective_max_tier = self._server_max_tier
        else:
            effective_max_tier = request.max_retrieval_tier

        for claim in claim_segments:
            claim_text = claim.claim_text
            decisive_report: Optional[ClaimReport] = None

            # -------------------------------------------------------------
            # Tier 1: In-Context Document Search (REQ-RET-05, ADR-0004)
            # -------------------------------------------------------------
            if effective_max_tier >= 1 and request.context_documents:
                t1_chunks = self._retrieval_adapter.search(
                    claim=claim_text,
                    tier=1,
                    context_docs=request.context_documents,
                    limit=self._top_k_per_tier,
                )
                decisive_report = self._evaluate_chunks_for_claim(claim, t1_chunks, tier=1)

            if decisive_report is not None:
                # Decisive Tier 1 evidence found: SHORT-CIRCUIT!
                claims_reports.append(decisive_report)
                continue

            # -------------------------------------------------------------
            # Tier 2: Local Hybrid Knowledge Base Search (REQ-RET-05, ADR-0004)
            # -------------------------------------------------------------
            if effective_max_tier >= 2:
                t2_chunks = self._retrieval_adapter.search(
                    claim=claim_text,
                    tier=2,
                    limit=self._top_k_per_tier,
                )
                decisive_report = self._evaluate_chunks_for_claim(claim, t2_chunks, tier=2)

            if decisive_report is not None:
                # Decisive Tier 2 evidence found!
                claims_reports.append(decisive_report)
                continue

            # -------------------------------------------------------------
            # Tier 3: External Web Search (REQ-RET-05, REQ-RET-06, ADR-0004)
            # Strict Opt-in: executes ONLY if client_max_tier == 3 AND SERVER_MAX_TIER >= 3
            # -------------------------------------------------------------
            if effective_max_tier >= 3 and request.max_retrieval_tier >= 3:
                t3_chunks = self._retrieval_adapter.search(
                    claim=claim_text,
                    tier=3,
                    limit=self._top_k_per_tier,
                    client_max_tier=request.max_retrieval_tier,
                )
                decisive_report = self._evaluate_chunks_for_claim(claim, t3_chunks, tier=3)

            if decisive_report is not None:
                # Decisive Tier 3 evidence found!
                claims_reports.append(decisive_report)
                continue

            # Inconclusive/fallback across all permitted tiers
            claims_reports.append(
                ClaimReport(
                    claim_id=claim.claim_id,
                    claim_text=claim_text,
                    original_sentence=claim.original_sentence,
                    state=ClaimState.UNVERIFIED,
                    probabilities={"contradiction": 0.0, "entailment": 0.0, "neutral": 1.0},
                    evidence_tier=None,
                    evidence_text=None,
                )
            )

        rates = compute_document_rates(claims_reports)
        is_safe, needs_review = evaluate_safety_gate(
            claims_reports,
            severe_contradiction_threshold=self._severe_contradiction_threshold,
        )

        initial_response = VerificationResponse(
            is_safe=is_safe,
            needs_review=needs_review,
            hallucination_rate=rates["hallucination_rate"],
            verification_rate=rates["verification_rate"],
            unverified_rate=rates["unverified_rate"],
            claims=claims_reports,
            sanitized_text=request.text,
        )

        if _is_secondary or not self._enable_mitigation:
            return initial_response

        if request.mitigation_mode == MitigationMode.FLAG:
            return initial_response

        if self._mitigation_pipeline is not None:
            return cast(
                VerificationResponse,
                self._mitigation_pipeline.execute(
                    request=request,
                    initial_response=initial_response,
                    raise_on_block=self._raise_on_block,
                ),
            )

        return initial_response


# Backward-compatible alias
VerificationOrchestrator = WaterfallOrchestrator


def get_orchestrator(
    store: Optional[DualIndexStore] = None,
    nli_verifier: Optional[BaseNLIVerifier] = None,
    retrieval_adapter: Optional[RetrievalAdapter] = None,
    segmenter: Optional[ClaimSegmenter] = None,
    use_mock: bool = False,
    rewriter: Optional[Any] = None,
    mitigation_pipeline: Optional[Any] = None,
    enable_mitigation: bool = True,
    raise_on_block: bool = True,
    **kwargs: Any,
) -> WaterfallOrchestrator:
    """Factory creating an instantiated WaterfallOrchestrator."""
    if store is None:
        if (
            retrieval_adapter is not None
            and hasattr(retrieval_adapter, "tier2_adapter")
            and hasattr(retrieval_adapter.tier2_adapter, "store")
        ):
            active_store = retrieval_adapter.tier2_adapter.store
        else:
            active_store = DualIndexStore(in_memory=True)
    else:
        active_store = store

    active_nli = nli_verifier
    if active_nli is None:
        active_nli = get_nli_verifier(use_mock=use_mock)

    active_rewriter = rewriter
    if active_rewriter is None and enable_mitigation and mitigation_pipeline is None:
        from src.mitigation.generator import get_generator
        from src.mitigation.rewriter import TargetedRewriter

        gen = get_generator(use_mock=use_mock)
        active_rewriter = TargetedRewriter(generator=gen)

    return WaterfallOrchestrator(
        store=active_store,
        nli_verifier=active_nli,
        retrieval_adapter=retrieval_adapter,
        segmenter=segmenter,
        rewriter=active_rewriter,
        mitigation_pipeline=mitigation_pipeline,
        enable_mitigation=enable_mitigation,
        raise_on_block=raise_on_block,
        **kwargs,
    )
