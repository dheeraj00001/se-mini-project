"""Two-Tier Evaluation Harness & Empirical Benchmark Suite.

Implements REQ-MIT-06, NFR-QUAL-02, NFR-QUAL-03, ADR-0009.
"""

from src.eval.metrics import (
    compute_classification_metrics,
    compute_retrieval_metrics,
)
from src.eval.calibration import (
    CalibrationResult,
    calibrate_nli_thresholds,
)
from src.eval.benchmark import (
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkSample,
    load_fever_dataset,
    load_halueval_dataset,
    run_tier_b_benchmark,
    split_calibration_test,
)

__all__ = [
    "compute_classification_metrics",
    "compute_retrieval_metrics",
    "CalibrationResult",
    "calibrate_nli_thresholds",
    "BenchmarkConfig",
    "BenchmarkReport",
    "BenchmarkSample",
    "load_fever_dataset",
    "load_halueval_dataset",
    "run_tier_b_benchmark",
    "split_calibration_test",
]
