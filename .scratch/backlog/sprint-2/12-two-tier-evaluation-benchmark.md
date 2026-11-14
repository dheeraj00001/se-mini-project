# 12 — Two-Tier Evaluation Harness & Empirical Benchmark Suite

**What to build:**
A comprehensive evaluation suite consisting of a 50-case local synthetic regression test suite in `pytest` (Tier A) and an offline benchmark runner over FEVER and HaluEval validation splits (Tier B) computing Precision, Recall, F1, Confusion Matrix, ROC-AUC, and separate retrieval metrics (Recall@K, MRR).

**Sprint:** Sprint 2  
**Owner:** Student 4 (Evaluation Lead) & Student 2 (NLI Calibration & Diagnostics)  
**Story Points:** 5 SP  
**SRS Reference:** REQ-MIT-06, NFR-QUAL-02, NFR-QUAL-03, ADR-0009  
**Blocked by:** 06 — NLI-Aware Waterfall Orchestration, 08 — Secondary Verification Loop, 10 — OpenAI Reverse Proxy  
**Status:** ready-for-agent  

### Context & Scope
To provide rigorous academic proof for UE24CS341A evaluation, this ticket establishes two evaluation tiers:
- **Tier A (CI/CD Regression)**: 50 curated edge-case tests validating the entire pipeline end-to-end in `pytest` on every commit.
- **Tier B (Empirical Benchmarking & Calibration)**: Evaluates FEVER (mapping directly to 3-way Supported/Refuted/NotEnoughInfo states) and HaluEval. Student 2 calibrates the 0.70/0.85 decision thresholds on a held-out calibration split, generating ROC curves, confusion matrices, and separate retrieval metrics (Recall@K, MRR) to isolate retrieval quality from NLI accuracy.

### Acceptance Criteria
- [ ] Tier A test suite (`tests/eval_local_50.py`) contains 50 verified cases covering true facts, inverted dates, swapped entities, fabricated citations, and epistemic hedges.
- [ ] Tier A tests execute in $\le 60\text{ s}$ inside GitHub Actions CI.
- [ ] Tier B evaluation script (`scripts/run_benchmark.py`) loads FEVER and HaluEval validation slices via HuggingFace `datasets`.
- [ ] Strict train/calibration/test split discipline: Threshold tuning is conducted strictly on the calibration set, with final metrics reported on the test set.
- [ ] Computes and exports formal classification metrics via `scikit-learn`: Confusion Matrix, Precision, Recall, F1-Score, and ROC-AUC curve.
- [ ] Computes separate retrieval metrics for Qdrant + BM25: Recall@5, Recall@10, and Mean Reciprocal Rank (MRR).
- [ ] Generates markdown/CSV summary tables and figures ready for inclusion in the final project report and 2-minute demo video.

### Definition of Done
Script `python scripts/run_benchmark.py` runs to completion, outputting benchmark tables, confusion matrices, and ROC plots stored in `reports/benchmark_results.json` and `reports/figures/`.
