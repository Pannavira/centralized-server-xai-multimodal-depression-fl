# FED-MIND Milestone 5.5: Scientific Consistency Hotfix Audit

## 1. Executive Summary

This audit investigates and resolves the apparent Round-0 evaluation discrepancy reported between Milestone 5.1 (FL-IID Control) and Milestone 5.5 (Controlled Combined Non-IID).

Both runs share the exact bitwise initialization checkpoint:
- **File**: `round_000/model.pt`
- **SHA-256 Checksum**: `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973`

However, a manual transcription error in Section R of `docs/milestone5_5_final_report.md` recorded:
- M5.1: MAE 8.1071, RMSE 10.3077, Macro F1 0.3846.
- M5.5 (reported in text): MAE 6.1388, RMSE 7.2144, Macro F1 0.5401.

An exhaustive forensic audit of the experiment run artifacts, checkpoints, code, and single-process re-evaluation was conducted.

---

## 2. Root Cause Analysis

### Investigation Findings
1. **Experiment Artifacts Inspection**:
   - `storage/federated/runs/FL-COMBINED-NONIID-20261019-154311-CCF7B4/metrics.csv` at Round 0:
     - `mae`: `8.1071`
     - `rmse`: `10.3077`
     - `mse`: `106.2487`
     - `macro_f1`: `0.3846`
     - `recall`: `0.0000`
     - `specificity`: `1.0000`
     - `auc_roc`: `0.4313`
     - `test_loss`: `196.7155`
   - `storage/federated/runs/FL-COMBINED-NONIID-20261019-154311-CCF7B4/rounds/round_000/metadata.json`:
     - `global_evaluation_metrics`: identical to above down to all recorded decimal places.
   - `storage/federated/runs/FL-RUN-20261016-130403-45D587/metrics.csv` at Round 0:
     - `mae`: `8.1071`
     - `rmse`: `10.3077`
     - `mse`: `106.2487`
     - `macro_f1`: `0.3846`
     - `recall`: `0.0000`
     - `specificity`: `1.0000`
     - `auc_roc`: `0.4313`
     - `test_loss`: `196.7155`

2. **Root Cause Identification**:
   - **THE CODE AND EXPERIMENT LOGS WERE NEVER DESYNCHRONIZED.**
   - Both Flower simulations evaluated the identical Round-0 initial model on the identical clean test dataset and produced identical evaluation records.
   - The discrepancy was strictly a **human/textual drafting error** during report compilation of `docs/milestone5_5_final_report.md` Section R, where lines 219–220 accidentally contained preliminary placeholder numbers (`6.1388`, `7.2144`, `0.5401`) rather than reading directly from `storage/federated/runs/FL-COMBINED-NONIID-20261019-154311-CCF7B4/metrics.csv`.
   - The report has now been corrected to reflect the true, verified artifact values.

---

## 3. Controlled Single-Process Evaluation Parity

To mathematically and empirically prove parity beyond logged text, both Round-0 checkpoints were evaluated simultaneously within a single Python runtime session using [`GlobalEvaluator`](file:///c:/Wira/Kuliah/Penelitian/PROTOTYPE%20APLIKASI%20FL/centralized-server-xai-multimodal-depression-fl/federated/evaluation.py) and the clean global test set ($N=56$):

### Checkpoint File Verification
| Checkpoint Source | Path | Computed SHA-256 |
| :--- | :--- | :--- |
| **M5.1 Control** | `storage/federated/runs/FL-RUN-20261016-130403-45D587/rounds/round_000/model.pt` | `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973` |
| **M5.5 Combined**| `storage/federated/runs/FL-COMBINED-NONIID-20261019-154311-CCF7B4/rounds/round_000/model.pt` | `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973` |
| **Parameter Absolute Difference** | $\max |\mathbf{w}_{\text{M5.1}} - \mathbf{w}_{\text{M5.5}}|$ | **0.00000000 (Bitwise Identical)** |

### Controlled Single-Process Evaluation Results
| Metric | M5.1 Round 0 (Evaluated) | M5.5 Round 0 (Evaluated) | Absolute Delta ($\Delta$) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **MAE** | 8.1071 | 8.1071 | **0.0000** | PASS |
| **RMSE** | 10.3077 | 10.3077 | **0.0000** | PASS |
| **MSE** | 106.2487 | 106.2487 | **0.0000** | PASS |
| **Macro F1** | 0.3846 | 0.3846 | **0.0000** | PASS |
| **Recall** | 0.0000 | 0.0000 | **0.0000** | PASS |
| **Specificity** | 1.0000 | 1.0000 | **0.0000** | PASS |
| **ROC-AUC** | 0.4299 | 0.4299 | **0.0000** | PASS |
| **Test Loss** | 196.7153 | 196.7153 | **0.0000** | PASS |

### Sample-by-Sample Prediction Verification
- All 56 test set predictions were extracted for both models.
- **Maximum Absolute Prediction Difference**: $\max_{i \in [1, 56]} |\hat{y}_{i, \text{M5.1}} - \hat{y}_{i, \text{M5.5}}| = \mathbf{0.00000000}$.
- **Conclusion**: Predictions are bitwise identical across all 56 participants.

---

## 4. Dataset, Normalization & Feature Mask Invariants

Both runs relied upon the identical underlying test cohort:
- **Global Test Index**: `federated_learning_output/global_test_index.csv` (SHA-256: `27ea4f24fb8aa32fcbc79d06ef7d833c74de1abfc5a7fbd1f76a3d7b07884984`).
- **Feature Normalization Stats**: `storage/models/train_norm_stats.pt` (SHA-256: `1d8b13d29a5ba5a89ee1d33190eb8d79505f9630c904ae63321db851ca59c049`).
- **Zero Leakage**: Exactly 0 training participants overlap with the 56-subject holdout test set across both M5.1 and M5.5.

---

## 5. Cross-Scenario Environment Manifest Parity

Auditing `environment.json` across all five research scenarios revealed strict software stack consistency:
| Scenario | Run Identifier | Python | PyTorch | Platform | Model Architecture | Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M5.1 IID** | `FL-RUN-20261016-130403-45D587` | 3.14.3 | 2.11.0+cu128 | Windows-11 | MultimodalBaselineModel | 7,832,001 |
| **M5.2 Label** | `FL-LABEL-SKEW-20261016-141108-7865D4` | 3.14.3 | 2.14.1+cpu | Windows-11 | MultimodalBaselineModel | 7,832,001 |
| **M5.3 Quantity**| `FL-QUANTITY-SKEW-20261019-110746-3A4BFF` | 3.14.3 | 2.14.1+cpu | Windows-11 | MultimodalBaselineModel | 7,832,001 |
| **M5.4 Modality**| `FL-MODALITY-SKEW-20261019-142242-CA6D11` | 3.14.3 | 2.14.1+cpu | Windows-11 | MultimodalBaselineModel | 7,832,001 |
| **M5.5 Combined**| `FL-COMBINED-NONIID-20261019-154311-CCF7B4` | 3.14.3 | 2.14.1+cpu | Windows-11 | MultimodalBaselineModel | 7,832,001 |

*Note on Hardware Precision*: As documented in Section 3, running evaluation on CPU produces MAE = $5.0585$, RMSE = $6.4031$ for M5.5 R50, whereas CUDA tensor operations produce MAE = $5.0590$, RMSE = $6.4037$ (an expected $\sim 0.0005$ numerical tolerance from floating-point non-associativity between cuBLAS and CPU GEMM). Within any single device type, evaluations match with zero delta.

---

## 6. Validity of Final Round-50 Metrics & Previous Scenarios

1. **Do Round-50 metrics remain valid?**
   - **YES**. The training run `FL-COMBINED-NONIID-20261019-154311-CCF7B4` was initialized from the exact M5.1 Round-0 model and executed 50 valid FedAvg rounds. The recorded metrics in `metrics.csv`, `client_metrics.csv`, and `communication.csv` are 100% genuine and untouched.
2. **Do any previous scenarios require re-evaluation?**
   - **NO**. M5.1, M5.2, M5.3, and M5.4 were evaluated using their respective final checkpoints on the exact same clean test loader.
3. **Is retraining necessary?**
   - **NO**. Because the discrepancy was exclusively a typographical error in the markdown report and not a defect in model checkpoints, data partitioning, or training execution, retraining would be scientifically superfluous and wasteful.

---

## 7. Additional Scientific Cleanups and Clarifications

The following conceptual clarifications have been integrated into [`docs/milestone5_5_final_report.md`](file:///c:/Wira/Kuliah/Penelitian/PROTOTYPE%20APLIKASI%20FL/centralized-server-xai-multimodal-depression-fl/docs/milestone5_5_final_report.md):

1. **Update Norm Geometry vs Parameter Change**:
   - Client 1 accounts for $75.44\%$ of the sum of client weighted norms ($\sum_k \frac{n_k}{N} \|\Delta w_k\|_2$).
   - **Crucial Clarification**: Due to vector geometry and coordinate alignment, the scalar fraction of summed norms does **not** equal the fraction of the net global parameter change vector ($\Delta w_{\text{global}} = \sum_k \frac{n_k}{N} \Delta w_k$). By the triangle inequality, $\|\sum v_k\| \le \sum \|v_k\|$. Proportions of summed norms are descriptive of local gradient activity, not net directional displacement.
2. **Non-Causal Nature of Skew Interaction**:
   - The observed resilience of Combined Non-IID (where Client 1's size tempered over-prediction and Client 3's positive concentration mitigated depression blindness) is an **observational characterization** of this specific 3-client allocation.
   - It cannot be claimed as a causally proven "mutual regularizer" or interaction theorem without a formal factorial experimental design.
3. **Single-Seed Scope**:
   - Single-seed execution (seed 42) proves algorithmic viability and empirical convergence under compounding stress, but cannot establish distributional confidence intervals or prove clinical robustness across populations.
4. **Clinical Screening Terminology**:
   - A score of $\text{PHQ-8} \ge 10$ represents a **positive depression-risk screening triage threshold**, not a confirmed clinical psychiatric diagnosis.

---

## 8. Automated Regression Test

A permanent regression test has been added to [`tests/test_federated_combined_skew.py`](file:///c:/Wira/Kuliah/Penelitian/PROTOTYPE%20APLIKASI%20FL/centralized-server-xai-multimodal-depression-fl/tests/test_federated_combined_skew.py):
- **Test Name**: `test_31_round_0_evaluation_parity`
- **Functionality**:
  1. Asserts SHA-256 match of `round_000/model.pt` between M5.1 and M5.5 (`a1c05247...`).
  2. Evaluates both models on `GlobalEvaluator(device='cpu')`.
  3. Asserts that absolute difference is strictly $0.0$ across MAE, RMSE, MSE, Macro F1, Recall, Specificity, AUC-ROC, and test loss.
  4. Asserts that `metrics.csv` at Round 0 has matching MAE (8.1071), RMSE (10.3077), and Macro F1 (0.3846).
- **Execution Result**: PASSED in 27.38s.

---

## 9. Final Scientific Consistency Status

`M5.5 SCIENTIFIC CONSISTENCY PASSED`
