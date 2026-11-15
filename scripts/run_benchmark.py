#!/usr/bin/env python3
"""CLI Runner for Tier B Empirical Benchmarking & NLI Calibration.

Implements REQ-MIT-06, NFR-QUAL-02, NFR-QUAL-03, ADR-0009.
Definition of Done:
Script `python scripts/run_benchmark.py` runs to completion, outputting benchmark
tables, confusion matrices, and ROC plots stored in `reports/benchmark_results.json`
and `reports/figures/`.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure mini-project root is on python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.benchmark import BenchmarkConfig, run_tier_b_benchmark  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for benchmark execution."""
    parser = argparse.ArgumentParser(
        description="Tier B Empirical Benchmark & NLI Threshold Calibration Suite"
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        default=False,
        help="Use fast MockNLIVerifier instead of downloading/running DeBERTa-v3 cross-encoder",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Pinned random seed for reproducibility (default: 42 per NFR-QUAL-03)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save benchmark reports and figures (default: reports)",
    )
    parser.add_argument(
        "--fever-samples",
        type=int,
        default=60,
        help="Number of FEVER benchmark samples to evaluate (default: 60)",
    )
    parser.add_argument(
        "--halueval-samples",
        type=int,
        default=60,
        help="Number of HaluEval benchmark samples to evaluate (default: 60)",
    )
    parser.add_argument(
        "--calibration-ratio",
        type=float,
        default=0.30,
        help="Proportion of samples allocated for threshold calibration (default: 0.30)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Disable matplotlib figure generation",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose debug logging",
    )
    return parser.parse_args()


def main() -> int:
    """Execute benchmark and print summary tables to terminal."""
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 80)
    print(" DEHALLUCINATION ADD-ON MODULE — TIER B EMPIRICAL BENCHMARK SUITE")
    print("=" * 80)
    print(f"Random Seed        : {args.seed} (pinned per NFR-QUAL-03)")
    print(f"Output Directory   : {args.output_dir}")
    mode_name = (
        "MockNLIVerifier (fast/headless)" if args.use_mock else "DeBERTa-v3 Cross-Encoder (CPU)"
    )
    print(f"Verifier Mode      : {mode_name}")
    cal_pct = args.calibration_ratio * 100
    test_pct = (1 - args.calibration_ratio) * 100
    print(f"Calibration Ratio  : {cal_pct:.0f}% Calibration / {test_pct:.0f}% Test")
    print("-" * 80)

    config = BenchmarkConfig(
        fever_samples=args.fever_samples,
        halueval_samples=args.halueval_samples,
        calibration_ratio=args.calibration_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
        use_mock_nli=args.use_mock,
        generate_plots=not args.no_plots,
    )

    try:
        report = run_tier_b_benchmark(config=config)
    except Exception as exc:
        logging.exception("Benchmark execution failed: %s", exc)
        print(f"\n[FATAL ERROR] Benchmark failed: {exc}", file=sys.stderr)
        return 1

    cal = report.calibration
    tm = report.test_metrics
    ret = report.retrieval_metrics
    pc = tm.get("per_class", {})

    print("\n" + "=" * 80)
    print(" 1. THRESHOLD CALIBRATION RESULTS (Held-Out Calibration Split)")
    print("=" * 80)
    print(f"Optimal Entailment Threshold (tau_e)    : {cal.optimal_entailment_threshold:.2f}")
    print(f"Optimal Contradiction Threshold (tau_c) : {cal.optimal_contradiction_threshold:.2f}")
    print(f"Severe Contradiction Circuit Breaker    : {cal.severe_contradiction_threshold:.2f}")
    print(f"Baseline Macro F1 (at 0.70 / 0.70)      : {cal.baseline_macro_f1:.4f}")
    print(f"Optimal Macro F1                        : {cal.optimal_macro_f1:.4f}")
    print(f"Macro F1 Gain                           : {cal.f1_gain:+.4f}")

    print("\n" + "=" * 80)
    print(" 2. TEST SET CLASSIFICATION METRICS (Held-Out Test Split)")
    print("=" * 80)
    print(f"Overall Accuracy    : {tm.get('accuracy', 0.0):.4f}")
    print(f"Macro Precision     : {tm.get('macro_precision', 0.0):.4f}")
    print(f"Macro Recall        : {tm.get('macro_recall', 0.0):.4f}")
    print(f"Macro F1-Score      : {tm.get('macro_f1', 0.0):.4f}")
    print(f"Weighted F1-Score   : {tm.get('weighted_f1', 0.0):.4f}")
    roc_dict = tm.get("roc_auc") or {}
    if roc_dict.get("multiclass_macro_ovr") is not None:
        print(f"Multiclass ROC-AUC  : {roc_dict.get('multiclass_macro_ovr', 0.0):.4f}")

    print("\n--- Per-Class Breakdown ---")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 63)
    for lbl in ["SUPPORTED", "CONTRADICTED", "UNVERIFIED"]:
        c = pc.get(lbl, {})
        print(
            f"{lbl:<15} {c.get('precision', 0.0):<12.4f} {c.get('recall', 0.0):<12.4f} "
            f"{c.get('f1', 0.0):<12.4f} {c.get('support', 0):<10}"
        )

    print("\n--- Confusion Matrix ---")
    cm = tm.get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    hdr = f"{'Ground Truth':<15} {'Pred SUPPORTED':<16} {'Pred CONTRADICTED':<18} {'Pred UNVERIFIED':<16}"  # noqa: E501
    print(hdr)
    print("-" * 65)
    for idx, lbl in enumerate(["SUPPORTED", "CONTRADICTED", "UNVERIFIED"]):
        row = cm[idx] if idx < len(cm) else [0, 0, 0]
        print(f"{lbl:<15} {row[0]:<16} {row[1]:<18} {row[2]:<16}")

    print("\n" + "=" * 80)
    print(" 3. HYBRID RETRIEVAL METRICS (Qdrant + BM25 RRF)")
    print("=" * 80)
    print(f"Recall@1  : {ret.get('recall@1', 0.0):.4f}")
    print(f"Recall@5  : {ret.get('recall@5', 0.0):.4f}")
    print(f"Recall@10 : {ret.get('recall@10', 0.0):.4f}")
    print(f"MRR       : {ret.get('mrr', 0.0):.4f}")

    print("\n" + "=" * 80)
    print(" 4. OUTPUT ARTIFACTS GENERATED")
    print("=" * 80)
    out_dir = Path(args.output_dir)
    print(f"Structured JSON Report : {out_dir / 'benchmark_results.json'}")
    print(f"Markdown Summary Table : {out_dir / 'benchmark_summary.md'}")
    for fig in report.figures_generated:
        print(f"Figure Artifact        : {fig}")
    print("=" * 80)
    print("[SUCCESS] Tier B Empirical Benchmark completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
