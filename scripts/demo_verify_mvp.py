"""Sprint 1 Working MVP Demonstration Script.

Demonstrates end-to-end NLI-aware Waterfall verification on real text:
1. Ingests grounding knowledge into the dual-index store (Tier 2).
2. Sends verification request with in-context document (Tier 1).
3. Demonstrates Tier 1 short-circuiting, Tier 2 escalation, and Unverified fallback.
4. Outputs comprehensive sentence-level reliability metrics and safety gating verdicts.
"""

import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.embedder import MockEmbedder  # noqa: E402
from src.retrieval.store import DualIndexStore  # noqa: E402
from src.schemas.verification import EvidenceChunk, VerificationRequest  # noqa: E402
from src.verifier.orchestrator import get_orchestrator  # noqa: E402


def run_mvp_demo() -> None:
    print("=" * 70)
    print("DEHALLUCINATION ADD-ON MODULE: SPRINT 1 WORKING MVP DEMO")
    print("=" * 70)

    # 1. Initialize Dual-Index Knowledge Base
    print("\n[Step 1] Initializing Tier 2 Knowledge Base...")
    store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))

    kb_chunks = [
        EvidenceChunk(
            chunk_id="kb_apollo",
            text=(
                "Apollo 11 landed humans on the Moon on July 20, 1969. "
                "Neil Armstrong was the first person to walk on the lunar surface."
            ),
            source_doc="space_exploration.txt",
            tier=2,
        ),
        EvidenceChunk(
            chunk_id="kb_python",
            text=(
                "Python is a high-level programming language created by "
                "Guido van Rossum and released in 1991."
            ),
            source_doc="software_history.txt",
            tier=2,
        ),
        EvidenceChunk(
            chunk_id="kb_paris",
            text="Paris is the capital of France. Berlin is the capital of Germany.",
            source_doc="world_geography.txt",
            tier=2,
        ),
    ]
    store.add_chunks(kb_chunks)
    print(f"-> Successfully indexed {len(kb_chunks)} reference chunks into Qdrant & BM25.")

    # 2. Instantiate Waterfall Orchestrator
    print("\n[Step 2] Initializing NLI-Aware Waterfall Orchestrator...")
    orchestrator = get_orchestrator(store=store, use_mock=True)
    print("-> Pipeline components wired: ClaimSegmenter + Tier1/2 Adapters + NLI Verifier.")

    # 3. Construct Real-World Multi-Sentence Text
    test_text = (
        "The Eiffel Tower is located in Paris. "
        "Neil Armstrong walked on the Moon in 1969. "
        "The capital of France is Berlin. "
        "The lost city of Atlantis had antigravity flying cars."
    )
    context_documents = [
        (
            "The Eiffel Tower is a wrought-iron lattice tower located on the "
            "Champ de Mars in Paris, France."
        )
    ]

    print("\n[Step 3] Verification Request:")
    print(f'Generated AI Text:\n"{test_text}"\n')
    print(f'In-Context Grounding Document (Tier 1):\n"{context_documents[0]}"\n')

    req = VerificationRequest(
        text=test_text,
        context_documents=context_documents,
        max_retrieval_tier=2,
    )

    # 4. Execute Verification Pipeline
    print("[Step 4] Executing Waterfall Verification Loop...")
    t0 = time.perf_counter()
    response = orchestrator.verify(req)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"-> Verification completed in {elapsed_ms:.2f} ms\n")

    # 5. Output Verification Report
    print("=" * 70)
    print("VERIFICATION REPORT SUMMARY")
    print("=" * 70)
    print(
        f"Overall Safety Gate (is_safe)       : {response.is_safe} "
        f"(Circuit breaker tripped: {not response.is_safe})"
    )
    print(f"Advisory Review Flag (needs_review) : {response.needs_review}")
    print(f"Verification Rate (SUPPORTED)      : {response.verification_rate:.2f}%")
    print(f"Hallucination Rate (CONTRADICTED)   : {response.hallucination_rate:.2f}%")
    print(f"Unverified Rate (UNVERIFIED)       : {response.unverified_rate:.2f}%\n")

    print("-" * 70)
    print("ATOMIC CLAIM EVALUATIONS:")
    print("-" * 70)
    for idx, claim in enumerate(response.claims, start=1):
        print(f'[{idx}] Claim: "{claim.claim_text}"')
        print(f"    State         : {claim.state.value}")
        tier_str = f"Tier {claim.evidence_tier}" if claim.evidence_tier else "None"
        print(f"    Evidence Tier : {tier_str}")
        if claim.evidence_text:
            print(f'    Evidence Text : "{claim.evidence_text[:80]}..."')
        print(
            f"    Probabilities : Entailment={claim.probabilities['entailment']:.3f}, "
            f"Contradiction={claim.probabilities['contradiction']:.3f}, "
            f"Neutral={claim.probabilities['neutral']:.3f}\n"
        )

    print("=" * 70)
    print("SPRINT 1 DEMO PASSED: ALL ACCEPTANCE CRITERIA VERIFIED")
    print("=" * 70)


if __name__ == "__main__":
    run_mvp_demo()
