"""Statistical Classification and Retrieval Metrics Engine.

Implements REQ-MIT-06, NFR-QUAL-02, NFR-QUAL-03, ADR-0009.
Calculates formal classification metrics via scikit-learn (Precision, Recall, F1,
Confusion Matrix, ROC-AUC) and ranking metrics (Recall@K, MRR) for hybrid retrieval.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

DEFAULT_CLASSIFICATION_LABELS: List[str] = [
    "SUPPORTED",
    "CONTRADICTED",
    "UNVERIFIED",
]


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    y_probs: Optional[Sequence[Union[Dict[str, float], Sequence[float]]]] = None,
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive academic classification metrics via scikit-learn.

    Args:
        y_true: Ground truth string labels.
        y_pred: Model predicted string labels.
        y_probs: Optional predicted class probability distributions. Either a sequence of
            dicts mapping label -> probability or a 2D matrix matching labels order.
        labels: Explicit label ordering. Defaults to ["SUPPORTED", "CONTRADICTED", "UNVERIFIED"].

    Returns:
        Structured metrics dictionary containing:
        - confusion_matrix: 2D list of shape (len(labels), len(labels))
        - labels: ordered label strings
        - accuracy: float [0.0, 1.0]
        - macro_precision, macro_recall, macro_f1: float
        - weighted_precision, weighted_recall, weighted_f1: float
        - micro_precision, micro_recall, micro_f1: float
        - per_class: Dict[label, {"precision", "recall", "f1", "support"}]
        - roc_auc: Optional dict with multiclass ovr roc-auc and binary contradiction roc-auc
        - roc_curves: Optional dict mapping label to {fpr: list, tpr: list, thresholds: list}
    """
    class_labels = list(labels) if labels is not None else list(DEFAULT_CLASSIFICATION_LABELS)

    if not y_true or not y_pred:
        cm_empty = [[0] * len(class_labels) for _ in range(len(class_labels))]
        empty_per_class: Dict[str, Dict[str, float]] = {
            lbl: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for lbl in class_labels
        }
        return {
            "confusion_matrix": cm_empty,
            "labels": class_labels,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "per_class": empty_per_class,
            "roc_auc": None,
            "roc_curves": None,
        }

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_list = cm.tolist()

    # 2. Overall Accuracy
    acc = float(accuracy_score(y_true, y_pred))

    # 3. Macro / Weighted / Micro Averages
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="macro", zero_division=0.0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="weighted", zero_division=0.0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="micro", zero_division=0.0
    )

    # 4. Per-class metrics
    p_class, r_class, f1_class, supp_class = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average=None, zero_division=0.0
    )

    per_class_dict: Dict[str, Dict[str, float]] = {}
    for idx, lbl in enumerate(class_labels):
        per_class_dict[lbl] = {
            "precision": round(float(p_class[idx]), 4),
            "recall": round(float(r_class[idx]), 4),
            "f1": round(float(f1_class[idx]), 4),
            "support": int(supp_class[idx]),
        }

    metrics: Dict[str, Any] = {
        "confusion_matrix": cm_list,
        "labels": class_labels,
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(p_weighted), 4),
        "weighted_recall": round(float(r_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "micro_precision": round(float(p_micro), 4),
        "micro_recall": round(float(r_micro), 4),
        "micro_f1": round(float(f1_micro), 4),
        "per_class": per_class_dict,
        "roc_auc": None,
        "roc_curves": None,
    }

    # 5. ROC-AUC and ROC curves (if probabilities provided)
    if y_probs is not None and len(y_probs) == len(y_true):
        # Convert y_probs to numpy matrix of shape (N, len(class_labels))
        prob_rows: List[List[float]] = []
        for row in y_probs:
            if isinstance(row, dict):
                # Map label names (handling case differences e.g. "contradiction" vs "CONTRADICTED")
                prob_row = []
                for lbl in class_labels:
                    val = row.get(lbl, row.get(lbl.lower(), 0.0))
                    prob_row.append(float(val))
                prob_rows.append(prob_row)
            else:
                prob_rows.append([float(x) for x in row])

        prob_matrix = np.array(prob_rows, dtype=np.float64)

        # Normalize row probabilities if needed
        row_sums = prob_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        prob_matrix = prob_matrix / row_sums

        roc_dict: Dict[str, Any] = {}
        curves_dict: Dict[str, Dict[str, List[float]]] = {}

        unique_true = set(y_true)
        if len(unique_true) >= 2:
            try:
                y_bin_all = label_binarize(y_true, classes=class_labels)
                if y_bin_all.ndim == 1:
                    y_bin_all = y_bin_all.reshape(-1, 1)

                valid_cols = [
                    col_idx
                    for col_idx in range(y_bin_all.shape[1])
                    if len(np.unique(y_bin_all[:, col_idx])) == 2
                ]
                if valid_cols:
                    macro_auc = roc_auc_score(
                        y_bin_all[:, valid_cols],
                        prob_matrix[:, valid_cols],
                        average="macro",
                    )
                    roc_dict["multiclass_macro_ovr"] = round(float(macro_auc), 4)

                    weighted_auc = roc_auc_score(
                        y_bin_all[:, valid_cols],
                        prob_matrix[:, valid_cols],
                        average="weighted",
                    )
                    roc_dict["multiclass_weighted_ovr"] = round(float(weighted_auc), 4)
                else:
                    roc_dict["multiclass_macro_ovr"] = None
                    roc_dict["multiclass_weighted_ovr"] = None
            except Exception:
                roc_dict["multiclass_macro_ovr"] = None
                roc_dict["multiclass_weighted_ovr"] = None

            # Binary ROC-AUC and curves per class
            for idx, lbl in enumerate(class_labels):
                y_bin = [1 if y == lbl else 0 for y in y_true]
                if sum(y_bin) > 0 and sum(y_bin) < len(y_bin):
                    try:
                        auc_val = roc_auc_score(y_bin, prob_matrix[:, idx])
                        roc_dict[f"auc_{lbl.lower()}"] = round(float(auc_val), 4)
                        fpr, tpr, thresholds = roc_curve(y_bin, prob_matrix[:, idx])
                        curves_dict[lbl] = {
                            "fpr": [round(float(x), 4) for x in fpr.tolist()],
                            "tpr": [round(float(x), 4) for x in tpr.tolist()],
                            "thresholds": [round(float(x), 4) for x in thresholds.tolist()],
                        }
                    except Exception:
                        roc_dict[f"auc_{lbl.lower()}"] = None
                else:
                    roc_dict[f"auc_{lbl.lower()}"] = None

        metrics["roc_auc"] = roc_dict if roc_dict else None
        metrics["roc_curves"] = curves_dict if curves_dict else None

    return metrics


def compute_retrieval_metrics(
    eval_items: Sequence[Dict[str, Any]],
    k_values: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Compute formal ranking metrics (Recall@K, MRR, MAP@K) for hybrid retrieval.

    Args:
        eval_items: Sequence of evaluation query dicts, each with:
            - "query": Query string or claim.
            - "expected_chunk_ids": List of relevant chunk IDs (or single string ID).
            - "retrieved_chunk_ids": Ranked list of retrieved chunk IDs from search.
        k_values: Cutoff values K for Recall@K (default: 1, 5, 10).

    Returns:
        Dict mapping metric names to calculated float values [0.0, 1.0], e.g.:
        - "recall@1": float
        - "recall@5": float
        - "recall@10": float
        - "mrr": float (Mean Reciprocal Rank)
        - "total_queries": int
    """
    if not eval_items:
        res: Dict[str, float] = {f"recall@{k}": 0.0 for k in k_values}
        res["mrr"] = 0.0
        res["total_queries"] = 0.0
        return res

    total_queries = len(eval_items)
    reciprocal_ranks: List[float] = []
    hits_at_k: Dict[int, int] = {k: 0 for k in k_values}

    for item in eval_items:
        raw_expected = item.get("expected_chunk_ids", [])
        if isinstance(raw_expected, str):
            expected_set = {raw_expected}
        else:
            expected_set = set(raw_expected)

        retrieved: List[str] = list(item.get("retrieved_chunk_ids", []))

        # 1. Reciprocal Rank
        found_rank: Optional[int] = None
        for rank_idx, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in expected_set:
                found_rank = rank_idx
                break

        if found_rank is not None:
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

        # 2. Recall@K
        for k in k_values:
            top_k_chunks = set(retrieved[:k])
            # Hit if at least one relevant chunk is within top-k
            if top_k_chunks.intersection(expected_set):
                hits_at_k[k] += 1

    results: Dict[str, float] = {
        f"recall@{k}": round(hits_at_k[k] / total_queries, 4) for k in k_values
    }
    results["mrr"] = round(float(np.mean(reciprocal_ranks)), 4)
    results["total_queries"] = float(total_queries)

    return results
