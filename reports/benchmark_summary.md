# Empirical Benchmark & Evaluation Report (Tier B)

**Run Timestamp:** `2026-11-15T14:47:27.047598`  
**Pinned Random Seed:** `42` (NFR-QUAL-03)  
**Total Samples:** 120 (36 Calibration / 84 Test)  
**Datasets:** FEVER (60), HaluEval (60)

---

## 1. NLI Threshold Calibration (Student 2)

Threshold calibration was conducted strictly on the held-out calibration set to prevent optimistic overfitting (ADR-0009):

| Threshold Parameter | Baseline | Calibrated Optimal | Impact / Role |
| :--- | :---: | :---: | :--- |
| **Entailment ($\tau_e$)** | `0.70` | `0.50` | Minimum confidence for `SUPPORTED` claim state |
| **Contradiction ($\tau_c$)** | `0.70` | `0.50` | Minimum confidence for `CONTRADICTED` claim state |
| **Severe Contradiction ($\tau_{severe}$)** | `0.85` | `0.85` | Safety Gate circuit breaker tripping `is_safe = False` |
| **Calibration Macro F1** | `0.7205` | `0.7205` | F1 Gain: `+0.0000` |

---

## 2. Test Classification Metrics (scikit-learn)

Final classification metrics evaluated on the held-out test split using calibrated:

| Metric | Baseline Thresholds (0.70 / 0.70) | Calibrated Thresholds | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | `0.6071` | `0.6071` | `+0.0000` |
| **Macro Precision** | `0.6893` | `0.6893` | `+0.0000` |
| **Macro Recall** | `0.6410` | `0.6410` | `+0.0000` |
| **Macro F1-Score** | `0.6076` | `0.6076` | `+0.0000` |
| **Weighted F1-Score** | `0.6045` | `0.6045` | `+0.0000` |
| **Multiclass ROC-AUC (OvR)** | `0.5000` | `0.5000` | `+0.0000` |

### Per-Class Detailed Breakdown

| Claim State | Precision | Recall | F1-Score | Support | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`SUPPORTED`** | `0.5806` | `0.5806` | `0.5806` | 31 | `0.5000` |
| **`CONTRADICTED`** | `1.0000` | `0.4375` | `0.6087` | 32 | `0.5000` |
| **`UNVERIFIED`** | `0.4872` | `0.9048` | `0.6333` | 21 | `0.5000` |

### Test Confusion Matrix

| Ground Truth \ Predicted | SUPPORTED | CONTRADICTED | UNVERIFIED |
| :--- | :---: | :---: | :---: |
| **`SUPPORTED`** | 18 | 0 | 13 |
| **`CONTRADICTED`** | 11 | 14 | 7 |
| **`UNVERIFIED`** | 2 | 0 | 19 |

---

## 3. Hybrid Retrieval Performance (Qdrant + BM25 RRF)

Evaluation of hybrid Reciprocal Rank Fusion ($k=60$) on ground-truth corpus:

| Retrieval Metric | Score | Acceptance Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Recall@1** | `0.5000` | $\ge 0.60$ | WARN |
| **Recall@5** | `1.0000` | $\ge 0.80$ | PASS |
| **Recall@10** | `1.0000` | $\ge 0.90$ | PASS |
| **Mean Reciprocal Rank (MRR)** | `0.7000` | $\ge 0.70$ | PASS |

---

## 4. Generated Diagnostic Figures

- `reports/figures/confusion_matrix.png`
- `reports/figures/roc_curves.png`
- `reports/figures/calibration_grid.png`
- `reports/figures/retrieval_performance.png`
