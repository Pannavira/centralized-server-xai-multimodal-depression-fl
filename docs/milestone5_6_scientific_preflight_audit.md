# FED-MIND — Milestone 5.6 Scientific Preflight Audit
**Forensic Audit of Round-0 Invariants, Checkpoint Checksums, and Historical Parity Across M5.1–M5.5**  
**Date:** 2026-10-20  
**Status:** **GATE A PASSED — SCIENTIFIC CONSISTENCY FULLY RESOLVED**

---

## 1. Executive Summary

Before beginning Milestone 5.6 cross-scenario multi-seed validation, a forensic preflight audit was conducted to resolve the perceived Round-0 evaluation discrepancy reported between Milestone 5.1 and Milestone 5.5.

The audit examined all five historical prototype runs (M5.1–M5.5, Seed 42):
- **M5.1 FL-IID**: `FL-RUN-20261016-130403-45D587`
- **M5.2 Label Skew**: `FL-LABEL-SKEW-20261016-141108-7865D4`
- **M5.3 Quantity Skew**: `FL-QUANTITY-SKEW-20261019-110746-3A4BFF`
- **M5.4 Modality Skew**: `FL-MODALITY-SKEW-20261019-142242-CA6D11`
- **M5.5 Combined Non-IID**: `FL-COMBINED-NONIID-20261019-154311-CCF7B4`

### Key Findings
1. **Bitwise Identical Initial Checkpoints**: All five Round-000 checkpoints (`round_000/model.pt`) share the identical byte size (`31,350,348` bytes) and the exact same SHA-256 checksum:
   `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973`.
2. **Canonical Evaluation Parity**: Re-evaluating all five Round-000 models simultaneously using `GlobalEvaluator` on the clean 56-subject global test cohort yields bitwise and numerically identical metrics down to $0.0000$ absolute delta:
   $$\text{MAE} = 8.1071, \quad \text{RMSE} = 10.3077, \quad \text{Macro F1} = 0.3846, \quad \text{Recall} = 0.0000, \quad \text{Specificity} = 1.0000$$
3. **Sample-by-Sample Prediction Verification**: The maximum pairwise difference in continuous predicted PHQ-8 scores across all $5 \times 56 = 280$ evaluations is strictly $\mathbf{0.00000000}$.
4. **Historical Experiment Logs (`metrics.csv`)**: All five historical `metrics.csv` files recorded the identical Round-0 values ($\text{MAE}=8.1071$, $\text{RMSE}=10.3077$, $\text{F1}=0.3846$).
5. **Root Cause**: The apparent discrepancy ($\text{MAE}=6.1388$ in M5.5) was strictly a **human drafting transcription error** in Section R of an early draft of `docs/milestone5_5_final_report.md`. No model checkpoint, code execution, or experimental artifact was ever desynchronized.
6. **Round-50 Integrity**: Re-evaluation of all five Round-50 checkpoints matches recorded values down to $0.0000$.

---

## 2. Checkpoint Verification Matrix

| Milestone & Scenario | Run Identifier | Round-000 Path | Checkpoint SHA-256 | Exact Byte Size | Strict Load |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M5.1 IID Control** | `FL-RUN-20261016-130403-45D587` | `rounds/round_000/model.pt` | `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973` | 31,350,348 | PASS |
| **M5.2 Label Skew** | `FL-LABEL-SKEW-20261016-141108-7865D4` | `rounds/round_000/model.pt` | `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973` | 31,350,348 | PASS |
| **M5.3 Quantity Skew**| `FL-QUANTITY-SKEW-20261019-110746-3A4BFF` | `rounds/round_000/model.pt` | `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973` | 31,350,348 | PASS |
| **M5.4 Modality Skew**| `FL-MODALITY-SKEW-20261019-142242-CA6D11` | `rounds/round_000/model.pt` | `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973` | 31,350,348 | PASS |
| **M5.5 Combined Skew**| `FL-COMBINED-NONIID-20261019-154311-CCF7B4` | `rounds/round_000/model.pt` | `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973` | 31,350,348 | PASS |

---

## 3. Dataset and Feature Normalization Invariants

The evaluation was performed against the authoritative immutable hold-out test set:

| Artifact | Canonical Path | SHA-256 Checksum | Invariant Property |
| :--- | :--- | :--- | :--- |
| **Global Hold-Out Test** | `federated_learning_output/global_test_index.csv` | `27ea4f24fb8aa32fcbc79d06ef7d833c74de1abfc5a7fbd1f76a3d7b07884984` | Exactly 56 subjects, 21 positive ($\ge 10$), 35 negative ($< 10$) |
| **Feature Normalization** | `storage/models/train_norm_stats.pt` | `1d8b13d29a5ba5a89ee1d33190eb8d79505f9630c904ae63321db851ca59c049` | Z-score scaling parameters derived strictly from training cohort |
| **Participant Leakage** | N/A | Zero | 0 training participants overlap with hold-out test or shared validation |

---

## 4. Canonical Single-Process Evaluation Results

All five Round-000 checkpoints and all five Round-050 checkpoints were evaluated sequentially using `GlobalEvaluator(device='cpu')`:

### 4.1 Round-000 Evaluation Comparison
| Metric | M5.1 IID | M5.2 Label | M5.3 Quantity | M5.4 Modality | M5.5 Combined | Max Delta ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test Loss** | 196.7155 | 196.7155 | 196.7155 | 196.7155 | 196.7155 | **0.0000** | IDENTICAL |
| **MAE** | 8.1071 | 8.1071 | 8.1071 | 8.1071 | 8.1071 | **0.0000** | IDENTICAL |
| **RMSE** | 10.3077 | 10.3077 | 10.3077 | 10.3077 | 10.3077 | **0.0000** | IDENTICAL |
| **MSE** | 106.2487 | 106.2487 | 106.2487 | 106.2487 | 106.2487 | **0.0000** | IDENTICAL |
| **Macro F1** | 0.3846 | 0.3846 | 0.3846 | 0.3846 | 0.3846 | **0.0000** | IDENTICAL |
| **Recall** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | IDENTICAL |
| **Specificity**| 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **0.0000** | IDENTICAL |
| **ROC-AUC** | 0.4313 | 0.4313 | 0.4313 | 0.4313 | 0.4313 | **0.0000** | IDENTICAL |
| **TP / FP / TN / FN** | 0 / 0 / 35 / 21 | 0 / 0 / 35 / 21 | 0 / 0 / 35 / 21 | 0 / 0 / 35 / 21 | 0 / 0 / 35 / 21 | **0 / 0 / 0 / 0** | IDENTICAL |

### 4.2 Pairwise Sample-Level Prediction Difference
$$\max_{i \in [1, 56]} |\hat{y}_{i, \text{run}_A} - \hat{y}_{i, \text{run}_B}|$$

| Comparison Pair | Max Absolute Difference |
| :--- | :--- |
| M5.1 vs M5.2 | **0.00000000** |
| M5.1 vs M5.3 | **0.00000000** |
| M5.1 vs M5.4 | **0.00000000** |
| M5.1 vs M5.5 | **0.00000000** |
| M5.2 vs M5.3 | **0.00000000** |
| M5.2 vs M5.4 | **0.00000000** |
| M5.2 vs M5.5 | **0.00000000** |
| M5.3 vs M5.4 | **0.00000000** |
| M5.3 vs M5.5 | **0.00000000** |
| M5.4 vs M5.5 | **0.00000000** |

---

## 5. Recorded vs Evaluated Checkpoint Metrics (Round 50)

| Scenario | Run ID | Recorded MAE | Evaluated MAE | Delta ($\Delta$) | Recorded F1 | Evaluated F1 | Delta ($\Delta$) | Round-050 Checkpoint SHA-256 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **M5.1 IID** | `FL-RUN-20261016-130403-45D587` | 4.9997 | 4.9997 | **0.0000** | 0.7005 | 0.7005 | **0.0000** | `41eab5039782720ed25e4723a74ce060450c6e2493dd877066eebc98334a1220` |
| **M5.2 Label** | `FL-LABEL-SKEW-20261016-141108-7865D4` | 4.6964 | 4.6964 | **0.0000** | 0.6725 | 0.6725 | **0.0000** | `583730ed878cc8527bc1cdf18e139294e4b8b45ac819e1f6f6edd46eb25f4ffd` |
| **M5.3 Quantity**| `FL-QUANTITY-SKEW-20261019-110746-3A4BFF` | 5.6170 | 5.6170 | **0.0000** | 0.5892 | 0.5892 | **0.0000** | `0019ce212413e8956edc71f96d6052fe6a27e374e7963b43c30f20e5b6006e68` |
| **M5.4 Modality**| `FL-MODALITY-SKEW-20261019-142242-CA6D11` | 4.9739 | 4.9739 | **0.0000** | 0.7214 | 0.7214 | **0.0000** | `abe2a3e18352f59256a296942eb08339bd71cb08e713827e5905a47963dcee79` |
| **M5.5 Combined**| `FL-COMBINED-NONIID-20261019-154311-CCF7B4` | 5.0585 | 5.0585 | **0.0000** | 0.6680 | 0.6680 | **0.0000** | `c3b41ab7363aba7f1517953326162165b4541b59ecdffd40099318e5364c87b3` |

---

## 6. Root-Cause Analysis and Assessment

1. **Was training corrupted or desynchronized?**
   **No.** All five runs used the verified initialization checkpoint `a1c052476e...` and completed 50 rounds with 3/3 clients participating.
2. **Was evaluation corrupted?**
   **No.** All five `metrics.csv` files contain the correct Round-0 metrics (`MAE=8.1071`, `RMSE=10.3077`, `Macro F1=0.3846`).
3. **What caused the reported numbers (`MAE=6.1388`)?**
   A clerical drafting error in Section R of `docs/milestone5_5_final_report.md` inadvertently pasted preliminary text placeholders. The true artifact values were already audited and documented in `docs/milestone5_5_scientific_consistency_audit.md`.
4. **Impact on Multi-Seed Validation:**
   Because all five seed-42 runs are verified to be 100% methodologically and empirically sound, they qualify as **`REUSED_VERIFIED`** for the Seed 42 column of the Milestone 5.6 matrix.

---

## 7. Gate A Conclusion and Audit Status

- Round-0 discrepancy: **RESOLVED (Clerical error; artifacts 100% bitwise consistent)**
- Canonical evaluator: **PASS (Bitwise prediction agreement $\Delta = 0.00000000$)**
- Final checkpoints: **VERIFIED (Bitwise match with recorded CSV logs)**
- Seed 42 prototype runs: **QUALIFIED AS `REUSED_VERIFIED`**

**`M5.6 SCIENTIFIC CONSISTENCY PASSED`**
