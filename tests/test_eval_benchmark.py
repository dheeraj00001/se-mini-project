"""Unit and Integration Tests for Tier B Benchmarking, Calibration, and Metrics.

Validates REQ-MIT-06, NFR-QUAL-02, NFR-QUAL-03, ADR-0009.
RTM Coverage:
- UT_MIT_06 / ST_EVL_01: Evaluation harness, scikit-learn metrics, threshold calibration,
  HuggingFace dataset loading with offline fallbacks, retrieval ranking metrics,
  figure generation, and CLI runner.
"""

import json
from pathlib import Path
import pytest

from src.eval.benchmark import (
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkSample,
    load_fever_dataset,
    load_halueval_dataset,
    run_retrieval_benchmark,
    run_tier_b_benchmark,
    split_calibration_test,
)
from src.eval.calibration import (
    CalibrationResult,
    CalibrationSample,
    calibrate_nli_thresholds,
)
from src.eval.metrics import (
    DEFAULT_CLASSIFICATION_LABELS,
    compute_classification_metrics,
    compute_retrieval_metrics,
)
from src.verifier.nli import MockNLIVerifier

# ==============================================================================
# 1. Classification Metrics Tests (src.eval.metrics)
# ==============================================================================


def test_compute_classification_metrics_empty() -> None:
    """Verify compute_classification_metrics returns safe zero-filled dict on empty input."""
    res = compute_classification_metrics([], [])
    assert res["accuracy"] == 0.0
    assert res["macro_f1"] == 0.0
    assert res["labels"] == DEFAULT_CLASSIFICATION_LABELS
    assert len(res["confusion_matrix"]) == 3
    assert res["roc_auc"] is None


def test_compute_classification_metrics_perfect_predictions() -> None:
    """Verify 100% precision, recall, and F1 on identical ground truth and predictions."""
    y_true = ["SUPPORTED", "CONTRADICTED", "UNVERIFIED", "SUPPORTED"]
    y_pred = ["SUPPORTED", "CONTRADICTED", "UNVERIFIED", "SUPPORTED"]
    y_probs = [
        {"SUPPORTED": 0.9, "CONTRADICTED": 0.05, "UNVERIFIED": 0.05},
        {"SUPPORTED": 0.05, "CONTRADICTED": 0.9, "UNVERIFIED": 0.05},
        {"SUPPORTED": 0.05, "CONTRADICTED": 0.05, "UNVERIFIED": 0.9},
        {"SUPPORTED": 0.85, "CONTRADICTED": 0.1, "UNVERIFIED": 0.05},
    ]

    res = compute_classification_metrics(y_true, y_pred, y_probs=y_probs)

    assert res["accuracy"] == 1.0
    assert res["macro_f1"] == 1.0
    assert res["macro_precision"] == 1.0
    assert res["macro_recall"] == 1.0
    assert res["per_class"]["SUPPORTED"]["f1"] == 1.0
    assert res["per_class"]["CONTRADICTED"]["f1"] == 1.0
    assert res["per_class"]["UNVERIFIED"]["f1"] == 1.0

    # ROC-AUC verification
    assert res["roc_auc"] is not None
    assert res["roc_auc"]["multiclass_macro_ovr"] == 1.0
    assert res["roc_curves"] is not None
    assert "SUPPORTED" in res["roc_curves"]


def test_compute_classification_metrics_with_errors() -> None:
    """Verify metrics calculation with mixed predictions and label list."""
    y_true = ["SUPPORTED", "SUPPORTED", "CONTRADICTED", "UNVERIFIED"]
    y_pred = ["SUPPORTED", "CONTRADICTED", "CONTRADICTED", "CONTRADICTED"]

    res = compute_classification_metrics(y_true, y_pred)

    assert res["accuracy"] == 0.50
    assert 0.0 < res["macro_f1"] < 1.0
    assert res["per_class"]["UNVERIFIED"]["recall"] == 0.0


def test_compute_classification_metrics_custom_labels_and_raw_probs() -> None:
    """Verify probability list input format and custom class labels."""
    labels = ["POS", "NEG"]
    y_true = ["POS", "NEG", "POS"]
    y_pred = ["POS", "NEG", "NEG"]
    y_probs = [
        [0.9, 0.1],
        [0.2, 0.8],
        [0.4, 0.6],
    ]

    res = compute_classification_metrics(y_true, y_pred, y_probs=y_probs, labels=labels)

    assert res["labels"] == labels
    assert res["accuracy"] == round(2.0 / 3.0, 4)
    assert res["roc_auc"] is not None


# ==============================================================================
# 2. Retrieval Metrics Tests (src.eval.metrics)
# ==============================================================================


def test_compute_retrieval_metrics_empty() -> None:
    """Verify compute_retrieval_metrics handles empty evaluation items safely."""
    res = compute_retrieval_metrics([], k_values=(1, 5, 10))
    assert res["recall@1"] == 0.0
    assert res["recall@5"] == 0.0
    assert res["recall@10"] == 0.0
    assert res["mrr"] == 0.0
    assert res["total_queries"] == 0.0


def test_compute_retrieval_metrics_accuracy() -> None:
    """Verify ranking metrics with exact expected values."""
    eval_items = [
        # Query 1: hit at rank 1
        {
            "query": "q1",
            "expected_chunk_ids": ["c1"],
            "retrieved_chunk_ids": ["c1", "c2", "c3"],
        },
        # Query 2: hit at rank 2
        {
            "query": "q2",
            "expected_chunk_ids": "c2",  # String instead of list
            "retrieved_chunk_ids": ["c1", "c2", "c3"],
        },
        # Query 3: hit at rank 3
        {
            "query": "q3",
            "expected_chunk_ids": ["c3", "c4"],
            "retrieved_chunk_ids": ["c1", "c2", "c3"],
        },
        # Query 4: miss
        {
            "query": "q4",
            "expected_chunk_ids": ["c99"],
            "retrieved_chunk_ids": ["c1", "c2", "c3"],
        },
    ]

    res = compute_retrieval_metrics(eval_items, k_values=(1, 2, 3))

    assert res["total_queries"] == 4.0
    # Recall@1: 1 hit out of 4 -> 0.25
    assert res["recall@1"] == 0.25
    # Recall@2: 2 hits out of 4 -> 0.50
    assert res["recall@2"] == 0.50
    # Recall@3: 3 hits out of 4 -> 0.75
    assert res["recall@3"] == 0.75
    # MRR: (1/1 + 1/2 + 1/3 + 0) / 4 = (1.0 + 0.5 + 0.3333 + 0) / 4 = 1.8333 / 4 = 0.4583
    assert pytest.approx(res["mrr"], abs=1e-3) == 0.4583


# ==============================================================================
# 3. NLI Threshold Calibration Tests (src.eval.calibration)
# ==============================================================================


def test_calibrate_nli_thresholds_empty() -> None:
    """Verify calibration returns safe default result on empty samples."""
    res = calibrate_nli_thresholds([])
    assert isinstance(res, CalibrationResult)
    assert res.optimal_entailment_threshold == 0.70
    assert res.optimal_contradiction_threshold == 0.70
    assert res.optimal_macro_f1 == 0.0


def test_calibrate_nli_thresholds_synthetic_optimization() -> None:
    """Verify grid search identifies optimal thresholds and computes F1 gain."""
    samples = [
        # Strongly supported
        CalibrationSample(
            claim_text="Paris is in France",
            premise_text="Paris is the capital of France",
            ground_truth="SUPPORTED",
            probabilities={"entailment": 0.85, "contradiction": 0.05, "neutral": 0.10},
        ),
        # Moderately supported (benefits from threshold <= 0.65)
        CalibrationSample(
            claim_text="Water boils at 100 degrees",
            premise_text="Water boils at 100 degrees Celsius",
            ground_truth="SUPPORTED",
            probabilities={"entailment": 0.65, "contradiction": 0.10, "neutral": 0.25},
        ),
        # Contradicted
        CalibrationSample(
            claim_text="Berlin is capital of France",
            premise_text="Paris is capital of France",
            ground_truth="CONTRADICTED",
            probabilities={"entailment": 0.05, "contradiction": 0.80, "neutral": 0.15},
        ),
        # Unverified (no premise)
        CalibrationSample(
            claim_text="Atlantis has flying cars",
            premise_text="",
            ground_truth="UNVERIFIED",
            probabilities={"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0},
        ),
    ]

    res = calibrate_nli_thresholds(
        samples,
        candidate_thresholds=[0.50, 0.60, 0.70, 0.80],
    )

    assert isinstance(res, CalibrationResult)
    assert 0.50 <= res.optimal_entailment_threshold <= 0.85
    assert 0.50 <= res.optimal_contradiction_threshold <= 0.85
    assert res.optimal_macro_f1 >= res.baseline_macro_f1
    assert len(res.sweep_results) == 16  # 4 x 4 grid


# ==============================================================================
# 4. Benchmark Dataset Loaders & Split Tests (src.eval.benchmark)
# ==============================================================================


def test_load_fever_dataset_curated_fallback() -> None:
    """Verify FEVER dataset loading returns valid BenchmarkSample instances."""
    samples = load_fever_dataset(max_samples=15)
    assert len(samples) == 15
    for s in samples:
        assert isinstance(s, BenchmarkSample)
        assert s.dataset_name == "FEVER"
        assert s.claim_text
        assert s.ground_truth in ["SUPPORTED", "CONTRADICTED", "UNVERIFIED"]


def test_load_halueval_dataset_curated_fallback() -> None:
    """Verify HaluEval dataset loading returns valid BenchmarkSample instances."""
    samples = load_halueval_dataset(max_samples=15)
    assert len(samples) == 15
    for s in samples:
        assert isinstance(s, BenchmarkSample)
        assert s.dataset_name == "HaluEval"
        assert s.claim_text
        assert s.ground_truth in ["SUPPORTED", "CONTRADICTED", "UNVERIFIED"]


def test_split_calibration_test_discipline() -> None:
    """Verify strict calibration/test split ratio and reproducibility (NFR-QUAL-03)."""
    fever = load_fever_dataset(max_samples=20)
    halu = load_halueval_dataset(max_samples=20)
    all_samples = fever + halu

    cal1, test1 = split_calibration_test(all_samples, calibration_ratio=0.30, seed=42)
    cal2, test2 = split_calibration_test(all_samples, calibration_ratio=0.30, seed=42)

    assert len(cal1) + len(test1) == len(all_samples)
    # Check 30% calibration ratio (+-1 sample)
    assert abs(len(cal1) - 12) <= 1
    # Check deterministic reproducibility under seed 42
    assert [s.sample_id for s in cal1] == [s.sample_id for s in cal2]
    assert [s.sample_id for s in test1] == [s.sample_id for s in test2]


def test_run_retrieval_benchmark() -> None:
    """Verify run_retrieval_benchmark evaluates Qdrant + BM25 ranking."""
    metrics = run_retrieval_benchmark(k_values=(1, 5, 10))

    assert "recall@1" in metrics
    assert "recall@5" in metrics
    assert "recall@10" in metrics
    assert "mrr" in metrics
    assert metrics["recall@10"] >= metrics["recall@5"] >= metrics["recall@1"]
    assert 0.0 <= metrics["mrr"] <= 1.0


# ==============================================================================
# 5. Full Tier B Benchmark & Artifact Generation Tests
# ==============================================================================


def test_run_tier_b_benchmark_end_to_end(tmp_path: Path) -> None:
    """Verify complete Tier B benchmark run produces JSON, markdown, and figure artifacts."""
    out_dir = tmp_path / "test_reports"
    config = BenchmarkConfig(
        fever_samples=12,
        halueval_samples=12,
        calibration_ratio=0.30,
        seed=42,
        output_dir=str(out_dir),
        use_mock_nli=True,
        generate_plots=True,
    )

    verifier = MockNLIVerifier()
    report = run_tier_b_benchmark(config=config, verifier=verifier)

    assert isinstance(report, BenchmarkReport)
    assert report.split_counts["total"] == 24
    assert report.calibration is not None
    assert report.test_metrics is not None
    assert report.retrieval_metrics is not None

    # Check generated files
    json_file = out_dir / "benchmark_results.json"
    assert json_file.is_file()
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["seed"] == 42
        assert "calibration" in data
        assert "test_metrics" in data

    md_file = out_dir / "benchmark_summary.md"
    assert md_file.is_file()
    md_content = md_file.read_text(encoding="utf-8")
    assert "# Empirical Benchmark & Evaluation Report (Tier B)" in md_content
    assert "## 1. NLI Threshold Calibration" in md_content
    assert "## 3. Hybrid Retrieval Performance" in md_content

    # Check figures
    figures_dir = out_dir / "figures"
    assert (figures_dir / "confusion_matrix.png").is_file()
    assert (figures_dir / "roc_curves.png").is_file()
    assert (figures_dir / "calibration_grid.png").is_file()
    assert (figures_dir / "retrieval_performance.png").is_file()


# ==============================================================================
# 6. CLI Runner Integration Test (scripts/run_benchmark.py)
# ==============================================================================


def test_cli_run_benchmark_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify scripts/run_benchmark.py executes via CLI main function with exit code 0."""
    from scripts.run_benchmark import main

    out_dir = tmp_path / "cli_reports"
    test_args = [
        "run_benchmark.py",
        "--use-mock",
        "--seed",
        "42",
        "--output-dir",
        str(out_dir),
        "--fever-samples",
        "10",
        "--halueval-samples",
        "10",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    exit_code = main()
    assert exit_code == 0
    assert (out_dir / "benchmark_results.json").is_file()
    assert (out_dir / "benchmark_summary.md").is_file()
