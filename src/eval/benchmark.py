"""Tier B Empirical Benchmarking, FEVER/HaluEval Evaluation & Report Generation.

Implements REQ-MIT-06, NFR-QUAL-02, NFR-QUAL-03, ADR-0009.
Roles: Student 4 (Evaluation Lead) & Student 2 (NLI Calibration & Diagnostics).

Executes offline empirical benchmarking:
1. Loads FEVER and HaluEval validation slices via HuggingFace datasets (with offline fallback).
2. Enforces strict train/calibration/test split discipline (pinned seed=42).
3. Calibrates decision thresholds on calibration split.
4. Computes formal classification metrics on held-out test split (scikit-learn).
5. Computes hybrid retrieval metrics for Qdrant + BM25 (Recall@K, MRR).
6. Generates JSON reports, markdown summaries, and matplotlib diagnostic figures.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import train_test_split

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
from src.schemas.verification import (
    ClaimState,
    EvidenceChunk,
    determine_claim_state_from_nli,
)
from src.verifier.nli import BaseNLIVerifier, get_nli_verifier

logger = logging.getLogger(__name__)


class BenchmarkSample(BaseModel):
    """A standardized evaluation sample for Tier B benchmarking."""

    model_config = ConfigDict(extra="ignore")

    sample_id: str
    dataset_name: str  # "FEVER" or "HaluEval"
    claim_text: str
    premise_text: str
    ground_truth: str  # "SUPPORTED", "CONTRADICTED", or "UNVERIFIED"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    """Configuration parameters for Tier B benchmark run."""

    model_config = ConfigDict(extra="forbid")

    fever_samples: int = Field(default=60, ge=10, description="Number of FEVER samples")
    halueval_samples: int = Field(default=60, ge=10, description="Number of HaluEval samples")
    calibration_ratio: float = Field(
        default=0.30, ge=0.1, le=0.5, description="Fraction of samples for calibration"
    )
    seed: int = Field(
        default=42, description="Pinned random seed for reproducibility (NFR-QUAL-03)"
    )
    output_dir: str = Field(
        default="reports", description="Output directory for reports and figures"
    )
    use_mock_nli: bool = Field(
        default=False, description="Use MockNLIVerifier for fast headless execution"
    )
    generate_plots: bool = Field(default=True, description="Generate matplotlib figure artifacts")


class BenchmarkReport(BaseModel):
    """Comprehensive serialized Tier B benchmark report."""

    model_config = ConfigDict(extra="ignore")

    timestamp: str
    seed: int
    datasets: Dict[str, int]
    split_counts: Dict[str, int]
    calibration: CalibrationResult
    test_metrics: Dict[str, Any]
    baseline_test_metrics: Dict[str, Any]
    retrieval_metrics: Dict[str, float]
    figures_generated: List[str]


# ==============================================================================
# Curated Data Loaders
# ==============================================================================


def _resolve_data_file(filename: str) -> Path:
    """Resolve path to a curated JSON benchmark file."""
    candidates = [
        Path.cwd() / "data" / "eval" / filename,
        Path(__file__).resolve().parent.parent.parent / "data" / "eval" / filename,
        Path(__file__).resolve().parent / "data" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _load_curated_data(filename: str) -> List[Dict[str, str]]:
    """Load curated benchmark records from decoupled JSON storage."""
    file_path = _resolve_data_file(filename)
    if not file_path.exists():
        logger.warning("Curated data file %s not found; returning empty list", file_path)
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _stratified_slice(
    rows: List[Dict[str, str]], max_samples: Optional[int]
) -> List[Dict[str, str]]:
    """Slice curated rows preserving representation across ground-truth classes."""
    if max_samples is None or max_samples >= len(rows):
        return rows

    by_label: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    balanced: List[Dict[str, str]] = []
    labels = sorted(list(by_label.keys()))
    idx = 0
    while len(balanced) < max_samples:
        added = False
        for lbl in labels:
            if idx < len(by_label[lbl]):
                balanced.append(by_label[lbl][idx])
                added = True
                if len(balanced) >= max_samples:
                    break
        if not added:
            break
        idx += 1
    return balanced


def load_fever_dataset(
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> List[BenchmarkSample]:
    """Load FEVER validation slice via HuggingFace datasets with curated offline fallback.

    Maps FEVER annotations directly to 3-way states:
    - "SUPPORTS" -> SUPPORTED
    - "REFUTES" -> CONTRADICTED
    - "NOT ENOUGH INFO" -> UNVERIFIED
    """
    samples: List[BenchmarkSample] = []

    try:
        from datasets import load_dataset

        logger.info("Attempting to load FEVER dataset from HuggingFace Hub...")
        hf_dataset = load_dataset("fever", "v1.0", split=split)
        label_map = {0: "SUPPORTED", 1: "CONTRADICTED", 2: "UNVERIFIED"}
        str_label_map = {
            "SUPPORTS": "SUPPORTED",
            "REFUTES": "CONTRADICTED",
            "NOT ENOUGH INFO": "UNVERIFIED",
        }

        count = 0
        limit = max_samples if max_samples is not None else 60
        for row in hf_dataset:
            claim = str(row.get("claim", "")).strip()
            raw_label = row.get("label")
            premise = str(row.get("evidence", "")).strip()
            if not premise:
                premise = str(row.get("evidence_sentence", "")).strip()

            if isinstance(raw_label, int) and raw_label in label_map:
                mapped_label = label_map[raw_label]
            elif isinstance(raw_label, str) and raw_label.upper() in str_label_map:
                mapped_label = str_label_map[raw_label.upper()]
            else:
                continue

            if claim:
                samples.append(
                    BenchmarkSample(
                        sample_id=f"fever_hf_{row.get('id', count)}",
                        dataset_name="FEVER",
                        claim_text=claim,
                        premise_text=premise,
                        ground_truth=mapped_label,
                        metadata={"split": split, "source": "huggingface"},
                    )
                )
                count += 1
                if count >= limit:
                    break

        if samples:
            logger.info("Successfully loaded %d FEVER samples from HuggingFace Hub", len(samples))
            return samples
    except Exception as exc:
        logger.info(
            "HuggingFace FEVER download unavailable (%s); using curated benchmark slice", exc
        )

    curated_rows = _load_curated_data("fever_curated.json")
    target_rows = _stratified_slice(curated_rows, max_samples)

    for row in target_rows:
        samples.append(
            BenchmarkSample(
                sample_id=row["id"],
                dataset_name="FEVER",
                claim_text=row["claim"],
                premise_text=row["premise"],
                ground_truth=row["label"],
                metadata={"split": split, "source": "curated_offline"},
            )
        )
    return samples


def load_halueval_dataset(
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> List[BenchmarkSample]:
    """Load HaluEval validation slice via HuggingFace datasets with curated offline fallback.

    Maps HaluEval annotations:
    - Factual grounded answer -> SUPPORTED
    - Hallucinated answer -> CONTRADICTED
    - Ungrounded / neutral answer -> UNVERIFIED
    """
    samples: List[BenchmarkSample] = []

    try:
        from datasets import load_dataset

        logger.info("Attempting to load HaluEval dataset from HuggingFace Hub...")
        hf_dataset = load_dataset("pku-alignment/HaluEval", "qa_data", split=split)

        count = 0
        limit = max_samples if max_samples is not None else 60
        for row in hf_dataset:
            knowledge = str(row.get("knowledge", "")).strip()
            right_ans = str(row.get("right_answer", "")).strip()
            hallu_ans = str(row.get("hallucinated_answer", "")).strip()

            if knowledge and right_ans:
                samples.append(
                    BenchmarkSample(
                        sample_id=f"halueval_hf_{count}_sup",
                        dataset_name="HaluEval",
                        claim_text=right_ans,
                        premise_text=knowledge,
                        ground_truth="SUPPORTED",
                        metadata={"source": "huggingface"},
                    )
                )
                count += 1

            if knowledge and hallu_ans and count < limit:
                samples.append(
                    BenchmarkSample(
                        sample_id=f"halueval_hf_{count}_con",
                        dataset_name="HaluEval",
                        claim_text=hallu_ans,
                        premise_text=knowledge,
                        ground_truth="CONTRADICTED",
                        metadata={"source": "huggingface"},
                    )
                )
                count += 1

            if count >= limit:
                break

        if samples:
            logger.info(
                "Successfully loaded %d HaluEval samples from HuggingFace Hub", len(samples)
            )
            return samples
    except Exception as exc:
        logger.info(
            "HuggingFace HaluEval download unavailable (%s); using curated benchmark slice", exc
        )

    curated_rows = _load_curated_data("halueval_curated.json")
    target_rows = _stratified_slice(curated_rows, max_samples)

    for row in target_rows:
        samples.append(
            BenchmarkSample(
                sample_id=row["id"],
                dataset_name="HaluEval",
                claim_text=row["claim"],
                premise_text=row["premise"],
                ground_truth=row["label"],
                metadata={"split": split, "source": "curated_offline"},
            )
        )
    return samples


def split_calibration_test(
    samples: Sequence[BenchmarkSample],
    calibration_ratio: float = 0.30,
    seed: int = 42,
) -> Tuple[List[BenchmarkSample], List[BenchmarkSample]]:
    """Enforce strict train/calibration/test split discipline with pinned seed (NFR-QUAL-03).

    Args:
        samples: Combined benchmark samples.
        calibration_ratio: Ratio for threshold calibration (default: 0.30).
        seed: Random seed for deterministic reproducibility.

    Returns:
        (calibration_samples, test_samples)
    """
    if len(samples) < 2:
        return list(samples), list(samples)

    labels = [s.ground_truth for s in samples]

    unique_labels, counts = np.unique(labels, return_counts=True)
    can_stratify = all(c >= 2 for c in counts) and len(unique_labels) >= 2

    cal_indices, test_indices = train_test_split(
        list(range(len(samples))),
        test_size=(1.0 - calibration_ratio),
        random_state=seed,
        stratify=labels if can_stratify else None,
    )

    cal_samples = [samples[i] for i in cal_indices]
    test_samples = [samples[i] for i in test_indices]

    return cal_samples, test_samples


# ==============================================================================
# Retrieval Benchmarking Engine (Qdrant + BM25)
# ==============================================================================


def run_retrieval_benchmark(
    k_values: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Evaluate Qdrant + BM25 hybrid retrieval against a ground truth corpus.

    Computes separate ranking metrics: Recall@1, Recall@5, Recall@10, and MRR.
    Isolates retrieval performance from NLI classification (ADR-0009).
    """
    from src.retrieval.embedder import MockEmbedder
    from src.retrieval.adapters import Tier2HybridAdapter
    from src.retrieval.store import DualIndexStore

    store = DualIndexStore(in_memory=True, embedder=MockEmbedder(dimension=384))

    benchmark_corpus = [
        (
            "c_space_01",
            "Apollo 11 landed humans on the Moon on July 20, 1969. Commander Neil Armstrong.",
        ),
        (
            "c_space_02",
            (
                "Mars is the fourth planet from the Sun and the second-smallest planet "
                "in Solar System."
            ),
        ),
        (
            "c_space_03",
            "The Hubble Space Telescope was launched into low Earth orbit in 1990.",
        ),
        (
            "c_geo_01",
            "Paris is the capital and most populous city of France, along the Seine River.",
        ),
        (
            "c_geo_02",
            "Mount Everest is Earth's highest mountain above sea level in the Himalayas.",
        ),
        (
            "c_geo_03",
            "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
        ),
        (
            "c_cs_01",
            "Python is a high-level programming language conceived by Guido van Rossum.",
        ),
        (
            "c_cs_02",
            "The Transformer architecture uses multi-head self-attention mechanisms for tokens.",
        ),
        (
            "c_cs_03",
            "FastAPI is a modern web framework for building REST APIs with Python type hints.",
        ),
        (
            "c_sci_01",
            "Water boils at 100 degrees Celsius and freezes at 0 degrees Celsius at sea level.",
        ),
        (
            "c_sci_02",
            "DNA consists of two strands that wind around each other to form a double helix.",
        ),
        (
            "c_sci_03",
            "Photosynthesis is the chemical process by which plants synthesize glucose from light.",
        ),
    ]

    chunks = [
        EvidenceChunk(
            chunk_id=cid,
            text=text,
            source_doc="benchmark_knowledge_corpus.txt",
            tier=2,
        )
        for cid, text in benchmark_corpus
    ]
    store.add_chunks(chunks)

    adapter = Tier2HybridAdapter(store=store)

    retrieval_queries = [
        {"query": "When did Apollo 11 land on the Moon?", "expected_chunk_ids": ["c_space_01"]},
        {"query": "What is the capital of France?", "expected_chunk_ids": ["c_geo_01"]},
        {
            "query": "Who created the Python programming language?",
            "expected_chunk_ids": ["c_cs_01"],
        },
        {
            "query": "What is the boiling point of water at sea level?",
            "expected_chunk_ids": ["c_sci_01"],
        },
        {
            "query": "What architecture uses self-attention mechanisms?",
            "expected_chunk_ids": ["c_cs_02"],
        },
        {
            "query": "What is the highest mountain on planet Earth?",
            "expected_chunk_ids": ["c_geo_02"],
        },
        {
            "query": "What molecule carries genetic instructions in a double helix?",
            "expected_chunk_ids": ["c_sci_03", "c_sci_02"],
        },
        {"query": "What is the largest ocean in the world?", "expected_chunk_ids": ["c_geo_03"]},
        {
            "query": "What space telescope was launched into low Earth orbit in 1990?",
            "expected_chunk_ids": ["c_space_03"],
        },
        {
            "query": "How do plants convert sunlight and carbon dioxide into glucose?",
            "expected_chunk_ids": ["c_sci_03"],
        },
    ]

    eval_items: List[Dict[str, Any]] = []
    for item in retrieval_queries:
        query_text = str(item["query"])
        retrieved_chunks = adapter.search(claim=query_text, limit=max(k_values))
        retrieved_ids = [c.chunk_id for c in retrieved_chunks]
        eval_items.append(
            {
                "query": query_text,
                "expected_chunk_ids": item["expected_chunk_ids"],
                "retrieved_chunk_ids": retrieved_ids,
            }
        )

    return compute_retrieval_metrics(eval_items, k_values=k_values)


# ==============================================================================
# Plot Generation (Matplotlib)
# ==============================================================================


def generate_benchmark_figures(
    test_metrics: Dict[str, Any],
    calibration: CalibrationResult,
    retrieval_metrics: Dict[str, float],
    output_dir: Path,
) -> List[str]:
    """Generate high-resolution matplotlib diagnostic plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_figures: List[str] = []

    # 1. Confusion Matrix Plot
    cm = test_metrics.get("confusion_matrix")
    labels = test_metrics.get("labels", DEFAULT_CLASSIFICATION_LABELS)
    if cm and len(cm) == len(labels):
        plt.figure(figsize=(6, 5), dpi=150)
        cm_arr = np.array(cm, dtype=int)
        plt.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(
            "NLI Classification Confusion Matrix\n(Held-out Test Split)",
            fontsize=12,
            fontweight="bold",
        )
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45, ha="right", fontsize=9)
        plt.yticks(tick_marks, labels, fontsize=9)

        thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 1.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = cm_arr[i, j]
                color = "white" if val > thresh else "black"
                plt.text(
                    j,
                    i,
                    format(val, "d"),
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                )

        plt.ylabel("Ground Truth Label", fontsize=10)
        plt.xlabel("Predicted Claim State", fontsize=10)
        plt.tight_layout()
        cm_path = output_dir / "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        generated_figures.append(str(cm_path))

    # 2. ROC Curves Plot
    curves = test_metrics.get("roc_curves")
    roc_auc = test_metrics.get("roc_auc") or {}
    if curves:
        plt.figure(figsize=(7, 5), dpi=150)
        colors = {"SUPPORTED": "green", "CONTRADICTED": "red", "UNVERIFIED": "orange"}
        for lbl, curve_data in curves.items():
            fpr = curve_data.get("fpr", [])
            tpr = curve_data.get("tpr", [])
            auc_val = roc_auc.get(f"auc_{lbl.lower()}", 0.0)
            if fpr and tpr:
                c = colors.get(lbl, "blue")
                plt.plot(fpr, tpr, color=c, lw=2, label=f"{lbl} (AUC = {auc_val:.3f})")

        plt.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=1.5,
            linestyle="--",
            label="Random Chance (0.50)",
        )
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
        plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
        plt.title("Multi-Class ROC Curves (One-vs-Rest)", fontsize=12, fontweight="bold")
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        roc_path = output_dir / "roc_curves.png"
        plt.savefig(roc_path)
        plt.close()
        generated_figures.append(str(roc_path))

    # 3. Threshold Calibration Surface Plot
    if calibration.sweep_results:
        plt.figure(figsize=(7, 4.5), dpi=150)
        sweep = calibration.sweep_results
        taus_c = sorted(list(set(pt["tau_contradiction"] for pt in sweep)))
        for tc in taus_c:
            pts = [pt for pt in sweep if pt["tau_contradiction"] == tc]
            pts.sort(key=lambda x: x["tau_entailment"])
            x_vals = [pt["tau_entailment"] for pt in pts]
            y_vals = [pt["macro_f1"] for pt in pts]
            plt.plot(
                x_vals,
                y_vals,
                marker="o",
                markersize=3,
                label=f"tau_c = {tc:.2f}",
                alpha=0.7,
            )

        plt.axvline(
            x=calibration.optimal_entailment_threshold,
            color="black",
            linestyle=":",
            label=f"Optimal tau_e ({calibration.optimal_entailment_threshold:.2f})",
        )
        plt.title(
            "NLI Decision Threshold Calibration Grid Search",
            fontsize=12,
            fontweight="bold",
        )
        plt.xlabel("Entailment Threshold (tau_e)", fontsize=10)
        plt.ylabel("Macro F1-Score", fontsize=10)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        cal_path = output_dir / "calibration_grid.png"
        plt.savefig(cal_path)
        plt.close()
        generated_figures.append(str(cal_path))

    # 4. Retrieval Ranking Performance Bar Chart
    plt.figure(figsize=(6, 4.5), dpi=150)
    ret_labels = ["Recall@1", "Recall@5", "Recall@10", "MRR"]
    ret_values = [
        retrieval_metrics.get("recall@1", 0.0),
        retrieval_metrics.get("recall@5", 0.0),
        retrieval_metrics.get("recall@10", 0.0),
        retrieval_metrics.get("mrr", 0.0),
    ]
    bar_colors = ["#2b5c8f", "#3c7cb8", "#4e9ce0", "#28a745"]
    bars = plt.bar(
        ret_labels,
        ret_values,
        color=bar_colors,
        width=0.55,
        edgecolor="black",
        linewidth=0.8,
    )
    plt.ylim(0.0, 1.15)
    plt.ylabel("Score [0.0 - 1.0]", fontsize=10)
    plt.title(
        "Hybrid Retrieval Performance (Qdrant + BM25 RRF)",
        fontsize=12,
        fontweight="bold",
    )
    plt.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, ret_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.03,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    plt.tight_layout()
    ret_path = output_dir / "retrieval_performance.png"
    plt.savefig(ret_path)
    plt.close()
    generated_figures.append(str(ret_path))

    return generated_figures


# ==============================================================================
# Markdown Summary Report Generator
# ==============================================================================


def generate_markdown_summary(report: BenchmarkReport) -> str:
    """Generate professional Markdown summary formatted for final project report."""
    cal = report.calibration
    tm = report.test_metrics
    bm = report.baseline_test_metrics
    ret = report.retrieval_metrics
    pc = tm.get("per_class", {})
    roc = tm.get("roc_auc") or {}

    total_s = report.split_counts.get("total", 0)
    cal_s = report.split_counts.get("calibration", 0)
    test_s = report.split_counts.get("test", 0)
    fever_c = report.datasets.get("FEVER", 0)
    halu_c = report.datasets.get("HaluEval", 0)

    lines: List[str] = [
        "# Empirical Benchmark & Evaluation Report (Tier B)",
        "",
        f"**Run Timestamp:** `{report.timestamp}`  ",
        f"**Pinned Random Seed:** `{report.seed}` (NFR-QUAL-03)  ",
        f"**Total Samples:** {total_s} ({cal_s} Calibration / {test_s} Test)  ",
        f"**Datasets:** FEVER ({fever_c}), HaluEval ({halu_c})",
        "",
        "---",
        "",
        "## 1. NLI Threshold Calibration (Student 2)",
        "",
        "Threshold calibration was conducted strictly on the held-out calibration set "
        "to prevent optimistic overfitting (ADR-0009):",
        "",
        "| Threshold Parameter | Baseline | Calibrated Optimal | Impact / Role |",
        "| :--- | :---: | :---: | :--- |",
        (
            f"| **Entailment ($\\tau_e$)** | `0.70` | "
            f"`{cal.optimal_entailment_threshold:.2f}` | "
            "Minimum confidence for `SUPPORTED` claim state |"
        ),
        (
            f"| **Contradiction ($\\tau_c$)** | `0.70` | "
            f"`{cal.optimal_contradiction_threshold:.2f}` | "
            "Minimum confidence for `CONTRADICTED` claim state |"
        ),
        (
            f"| **Severe Contradiction ($\\tau_{{severe}}$)** | `0.85` | "
            f"`{cal.severe_contradiction_threshold:.2f}` | "
            "Safety Gate circuit breaker tripping `is_safe = False` |"
        ),
        (
            f"| **Calibration Macro F1** | `{cal.baseline_macro_f1:.4f}` | "
            f"`{cal.optimal_macro_f1:.4f}` | F1 Gain: `{cal.f1_gain:+.4f}` |"
        ),
        "",
        "---",
        "",
        "## 2. Test Classification Metrics (scikit-learn)",
        "",
        "Final classification metrics evaluated on the held-out test split using calibrated:",
        "",
        "| Metric | Baseline Thresholds (0.70 / 0.70) | Calibrated Thresholds | Delta |",
        "| :--- | :---: | :---: | :---: |",
    ]

    bm_auc = (bm.get("roc_auc") or {}).get("multiclass_macro_ovr")
    tm_auc = (roc or {}).get("multiclass_macro_ovr")
    bm_auc_str = f"{bm_auc:.4f}" if isinstance(bm_auc, (int, float)) else "N/A"
    tm_auc_str = f"{tm_auc:.4f}" if isinstance(tm_auc, (int, float)) else "N/A"
    delta_auc_str = (
        f"{(tm_auc - bm_auc):+.4f}"
        if isinstance(bm_auc, (int, float)) and isinstance(tm_auc, (int, float))
        else "N/A"
    )

    acc_diff = tm.get("accuracy", 0.0) - bm.get("accuracy", 0.0)
    prec_diff = tm.get("macro_precision", 0.0) - bm.get("macro_precision", 0.0)
    rec_diff = tm.get("macro_recall", 0.0) - bm.get("macro_recall", 0.0)
    f1_diff = tm.get("macro_f1", 0.0) - bm.get("macro_f1", 0.0)
    wf1_diff = tm.get("weighted_f1", 0.0) - bm.get("weighted_f1", 0.0)

    lines.extend(
        [
            (
                f"| **Accuracy** | `{bm.get('accuracy', 0.0):.4f}` | "
                f"`{tm.get('accuracy', 0.0):.4f}` | `{acc_diff:+.4f}` |"
            ),
            (
                f"| **Macro Precision** | `{bm.get('macro_precision', 0.0):.4f}` | "
                f"`{tm.get('macro_precision', 0.0):.4f}` | `{prec_diff:+.4f}` |"
            ),
            (
                f"| **Macro Recall** | `{bm.get('macro_recall', 0.0):.4f}` | "
                f"`{tm.get('macro_recall', 0.0):.4f}` | `{rec_diff:+.4f}` |"
            ),
            (
                f"| **Macro F1-Score** | `{bm.get('macro_f1', 0.0):.4f}` | "
                f"`{tm.get('macro_f1', 0.0):.4f}` | `{f1_diff:+.4f}` |"
            ),
            (
                f"| **Weighted F1-Score** | `{bm.get('weighted_f1', 0.0):.4f}` | "
                f"`{tm.get('weighted_f1', 0.0):.4f}` | `{wf1_diff:+.4f}` |"
            ),
            (
                f"| **Multiclass ROC-AUC (OvR)** | `{bm_auc_str}` | "
                f"`{tm_auc_str}` | `{delta_auc_str}` |"
            ),
            "",
            "### Per-Class Detailed Breakdown",
            "",
            "| Claim State | Precision | Recall | F1-Score | Support | ROC-AUC |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for lbl in DEFAULT_CLASSIFICATION_LABELS:
        c_data = pc.get(lbl, {})
        auc_val = roc.get(f"auc_{lbl.lower()}", "N/A")
        auc_str = f"{auc_val:.4f}" if isinstance(auc_val, float) else "N/A"
        lines.append(
            f"| **`{lbl}`** | `{c_data.get('precision', 0.0):.4f}` | "
            f"`{c_data.get('recall', 0.0):.4f}` | `{c_data.get('f1', 0.0):.4f}` | "
            f"{c_data.get('support', 0)} | `{auc_str}` |"
        )

    lines.extend(
        [
            "",
            "### Test Confusion Matrix",
            "",
            "| Ground Truth \\ Predicted | SUPPORTED | CONTRADICTED | UNVERIFIED |",
            "| :--- | :---: | :---: | :---: |",
        ]
    )

    cm = tm.get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for idx, lbl in enumerate(DEFAULT_CLASSIFICATION_LABELS):
        row = cm[idx] if idx < len(cm) else [0, 0, 0]
        lines.append(f"| **`{lbl}`** | {row[0]} | {row[1]} | {row[2]} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Hybrid Retrieval Performance (Qdrant + BM25 RRF)",
            "",
            "Evaluation of hybrid Reciprocal Rank Fusion ($k=60$) on ground-truth corpus:",
            "",
            "| Retrieval Metric | Score | Acceptance Threshold | Status |",
            "| :--- | :---: | :---: | :---: |",
            (
                f"| **Recall@1** | `{ret.get('recall@1', 0.0):.4f}` | $\\ge 0.60$ | "
                f"{'PASS' if ret.get('recall@1', 0.0) >= 0.60 else 'WARN'} |"
            ),
            (
                f"| **Recall@5** | `{ret.get('recall@5', 0.0):.4f}` | $\\ge 0.80$ | "
                f"{'PASS' if ret.get('recall@5', 0.0) >= 0.80 else 'WARN'} |"
            ),
            (
                f"| **Recall@10** | `{ret.get('recall@10', 0.0):.4f}` | $\\ge 0.90$ | "
                f"{'PASS' if ret.get('recall@10', 0.0) >= 0.90 else 'WARN'} |"
            ),
            (
                f"| **Mean Reciprocal Rank (MRR)** | `{ret.get('mrr', 0.0):.4f}` | $\\ge 0.70$ | "
                f"{'PASS' if ret.get('mrr', 0.0) >= 0.70 else 'WARN'} |"
            ),
            "",
            "---",
            "",
            "## 4. Generated Diagnostic Figures",
            "",
        ]
    )

    for fig in report.figures_generated:
        p = Path(fig)
        lines.append(f"- `reports/figures/{p.name}`")

    lines.append("")
    return "\n".join(lines)


# ==============================================================================
# Full Benchmark Orchestrator
# ==============================================================================


def run_tier_b_benchmark(
    config: Optional[BenchmarkConfig] = None,
    verifier: Optional[BaseNLIVerifier] = None,
) -> BenchmarkReport:
    """Execute complete Tier B empirical benchmark suite."""
    cfg = config if config is not None else BenchmarkConfig()
    output_path = Path(cfg.output_dir)
    figures_dir = output_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load benchmark datasets (FEVER and HaluEval)
    fever_samples = load_fever_dataset(max_samples=cfg.fever_samples)
    halueval_samples = load_halueval_dataset(max_samples=cfg.halueval_samples)
    all_samples = fever_samples + halueval_samples

    # 2. Strict calibration/test split with pinned seed (NFR-QUAL-03)
    cal_samples, test_samples = split_calibration_test(
        all_samples,
        calibration_ratio=cfg.calibration_ratio,
        seed=cfg.seed,
    )

    # 3. Instantiate NLI Verifier
    active_verifier: BaseNLIVerifier
    if verifier is not None:
        active_verifier = verifier
    else:
        active_verifier = get_nli_verifier(use_mock=cfg.use_mock_nli)

    # 4. Predict probabilities across calibration set
    cal_pairs = [(s.premise_text, s.claim_text) for s in cal_samples]
    cal_probs_list = active_verifier.predict_proba(cal_pairs)

    cal_items: List[CalibrationSample] = []
    for s, probs in zip(cal_samples, cal_probs_list):
        cal_items.append(
            CalibrationSample(
                claim_text=s.claim_text,
                premise_text=s.premise_text,
                ground_truth=s.ground_truth,
                probabilities=probs,
            )
        )

    # 5. Calibrate thresholds strictly on calibration set (Student 2)
    calibration_result = calibrate_nli_thresholds(
        calibration_samples=cal_items,
        candidate_thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
    )

    # 6. Evaluate on held-out test split using BOTH baseline and calibrated thresholds
    test_pairs = [(s.premise_text, s.claim_text) for s in test_samples]
    test_probs_list = active_verifier.predict_proba(test_pairs)
    test_true = [s.ground_truth for s in test_samples]

    # Baseline predictions (0.70 / 0.70)
    baseline_preds: List[str] = []
    for s, probs in zip(test_samples, test_probs_list):
        if not s.premise_text.strip():
            baseline_preds.append(ClaimState.UNVERIFIED.value)
        else:
            state = determine_claim_state_from_nli(
                probabilities=probs,
                entailment_threshold=0.70,
                contradiction_threshold=0.70,
            )
            baseline_preds.append(state.value)

    baseline_test_metrics = compute_classification_metrics(
        test_true, baseline_preds, y_probs=test_probs_list
    )

    # Calibrated optimal predictions
    calibrated_preds: List[str] = []
    for s, probs in zip(test_samples, test_probs_list):
        if not s.premise_text.strip():
            calibrated_preds.append(ClaimState.UNVERIFIED.value)
        else:
            state = determine_claim_state_from_nli(
                probabilities=probs,
                entailment_threshold=calibration_result.optimal_entailment_threshold,
                contradiction_threshold=calibration_result.optimal_contradiction_threshold,
            )
            calibrated_preds.append(state.value)

    test_metrics = compute_classification_metrics(
        test_true, calibrated_preds, y_probs=test_probs_list
    )

    # 7. Evaluate retrieval metrics for Qdrant + BM25
    retrieval_metrics = run_retrieval_benchmark(k_values=(1, 5, 10))

    # 8. Generate diagnostic figures
    generated_figures: List[str] = []
    if cfg.generate_plots:
        generated_figures = generate_benchmark_figures(
            test_metrics=test_metrics,
            calibration=calibration_result,
            retrieval_metrics=retrieval_metrics,
            output_dir=figures_dir,
        )

    # 9. Assemble BenchmarkReport
    report = BenchmarkReport(
        timestamp=datetime.now().isoformat(),
        seed=cfg.seed,
        datasets={
            "FEVER": len(fever_samples),
            "HaluEval": len(halueval_samples),
        },
        split_counts={
            "total": len(all_samples),
            "calibration": len(cal_samples),
            "test": len(test_samples),
        },
        calibration=calibration_result,
        test_metrics=test_metrics,
        baseline_test_metrics=baseline_test_metrics,
        retrieval_metrics=retrieval_metrics,
        figures_generated=generated_figures,
    )

    # 10. Persist reports to disk
    json_path = output_path / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    md_summary = generate_markdown_summary(report)
    md_path = output_path / "benchmark_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_summary)

    logger.info("Saved benchmark report to %s and %s", json_path, md_path)
    return report
