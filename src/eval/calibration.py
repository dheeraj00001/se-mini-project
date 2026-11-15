"""NLI Decision Threshold Calibration and Safety Diagnostics.

Implements REQ-MIT-06, NFR-QUAL-02, NFR-QUAL-03, ADR-0009.
Role: Student 2 (NLI Calibration & Diagnostics).

Enforces strict train/calibration/test split discipline:
Threshold tuning is conducted strictly on the calibration set,
with final metrics evaluated on the held-out test set.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, ConfigDict, Field

from src.eval.metrics import compute_classification_metrics
from src.schemas.verification import ClaimState, determine_claim_state_from_nli


@dataclass
class CalibrationSample:
    """A single evaluation sample for NLI threshold calibration."""

    claim_text: str
    premise_text: str
    ground_truth: str  # "SUPPORTED", "CONTRADICTED", or "UNVERIFIED"
    probabilities: Dict[str, float]


class CalibrationResult(BaseModel):
    """Calibrated decision thresholds and diagnostic statistics."""

    model_config = ConfigDict(extra="forbid")

    optimal_entailment_threshold: float = Field(
        default=0.70,
        description="Calibrated entailment decision threshold tau_e",
    )
    optimal_contradiction_threshold: float = Field(
        default=0.70,
        description="Calibrated contradiction decision threshold tau_c",
    )
    severe_contradiction_threshold: float = Field(
        default=0.85,
        description="Calibrated severe contradiction safety circuit breaker threshold",
    )
    baseline_macro_f1: float = Field(
        default=0.0,
        description="Macro F1 at baseline thresholds (0.70 / 0.70)",
    )
    optimal_macro_f1: float = Field(
        default=0.0,
        description="Macro F1 at optimal calibrated thresholds",
    )
    f1_gain: float = Field(
        default=0.0,
        description="Absolute gain in Macro F1 over baseline",
    )
    calibration_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed classification metrics on calibration split at optimal thresholds",
    )
    sweep_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Grid search evaluation points across candidate thresholds",
    )


def calibrate_nli_thresholds(
    calibration_samples: Sequence[CalibrationSample],
    candidate_thresholds: Optional[Sequence[float]] = None,
    severe_threshold: float = 0.85,
    metric_objective: str = "macro_f1",
) -> CalibrationResult:
    """Calibrate decision thresholds strictly on the held-out calibration split.

    Args:
        calibration_samples: Dedicated calibration split samples with probabilities and true labels.
        candidate_thresholds: Grid of threshold values to evaluate (default: 0.50 to 0.85 by 0.05).
        severe_threshold: Safety gate severe contradiction threshold (default: 0.85).
        metric_objective: Metric to maximize ("macro_f1" or "contradiction_f1").

    Returns:
        CalibrationResult containing optimal thresholds and grid search diagnostics.
    """
    if candidate_thresholds is None:
        candidate_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    if not calibration_samples:
        return CalibrationResult(
            optimal_entailment_threshold=0.70,
            optimal_contradiction_threshold=0.70,
            severe_contradiction_threshold=severe_threshold,
            baseline_macro_f1=0.0,
            optimal_macro_f1=0.0,
            f1_gain=0.0,
            calibration_metrics={},
            sweep_results=[],
        )

    y_true = [s.ground_truth.upper() for s in calibration_samples]
    y_probs = [s.probabilities for s in calibration_samples]

    # Evaluate baseline at (0.70, 0.70)
    baseline_preds: List[str] = []
    for s in calibration_samples:
        pred_state = determine_claim_state_from_nli(
            probabilities=s.probabilities,
            entailment_threshold=0.70,
            contradiction_threshold=0.70,
        )
        baseline_preds.append(pred_state.value)

    baseline_metrics = compute_classification_metrics(y_true, baseline_preds, y_probs=y_probs)
    baseline_f1 = float(baseline_metrics.get("macro_f1", 0.0))

    # Grid search across candidate thresholds
    best_score = -1.0
    best_tau_e = 0.70
    best_tau_c = 0.70
    sweep_results: List[Dict[str, Any]] = []

    for tau_c in candidate_thresholds:
        for tau_e in candidate_thresholds:
            preds: List[str] = []
            for s in calibration_samples:
                # If premise text is empty/whitespace, state is UNVERIFIED
                if not s.premise_text.strip():
                    preds.append(ClaimState.UNVERIFIED.value)
                else:
                    pred_state = determine_claim_state_from_nli(
                        probabilities=s.probabilities,
                        entailment_threshold=tau_e,
                        contradiction_threshold=tau_c,
                    )
                    preds.append(pred_state.value)

            eval_res = compute_classification_metrics(y_true, preds, y_probs=None)
            macro_f1 = float(eval_res.get("macro_f1", 0.0))
            contra_f1 = float(eval_res.get("per_class", {}).get("CONTRADICTED", {}).get("f1", 0.0))

            score = contra_f1 if metric_objective == "contradiction_f1" else macro_f1

            sweep_point = {
                "tau_entailment": round(tau_e, 2),
                "tau_contradiction": round(tau_c, 2),
                "macro_f1": round(macro_f1, 4),
                "contradiction_f1": round(contra_f1, 4),
                "accuracy": round(float(eval_res.get("accuracy", 0.0)), 4),
            }
            sweep_results.append(sweep_point)

            if score > best_score:
                best_score = score
                best_tau_e = tau_e
                best_tau_c = tau_c

    # Compute detailed metrics on calibration set using optimal thresholds
    optimal_preds: List[str] = []
    for s in calibration_samples:
        if not s.premise_text.strip():
            optimal_preds.append(ClaimState.UNVERIFIED.value)
        else:
            p_state = determine_claim_state_from_nli(
                probabilities=s.probabilities,
                entailment_threshold=best_tau_e,
                contradiction_threshold=best_tau_c,
            )
            optimal_preds.append(p_state.value)

    optimal_metrics = compute_classification_metrics(y_true, optimal_preds, y_probs=y_probs)
    optimal_f1 = float(optimal_metrics.get("macro_f1", 0.0))

    return CalibrationResult(
        optimal_entailment_threshold=round(best_tau_e, 2),
        optimal_contradiction_threshold=round(best_tau_c, 2),
        severe_contradiction_threshold=round(severe_threshold, 2),
        baseline_macro_f1=round(baseline_f1, 4),
        optimal_macro_f1=round(optimal_f1, 4),
        f1_gain=round(optimal_f1 - baseline_f1, 4),
        calibration_metrics=optimal_metrics,
        sweep_results=sweep_results,
    )
