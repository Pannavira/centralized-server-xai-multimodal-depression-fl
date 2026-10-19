# FED-MIND — Milestone 5.3: Pre-Flight Verification Report
**Controlled Non-IID Quantity Skew Experiment**
**Date:** 2026-10-19  
**Status:** PASSED (All pre-flight gates verified)

---

## 1. Executive Summary

This document certifies that the pre-flight verification for **Milestone 5.3: Controlled Non-IID Quantity Skew Experiment** has been completed. All baseline control artifacts from Milestone 5.1 (FL-IID Control) have been audited, cryptographic checksums verified, test suites executed, and environment integrity frozen.

---

## 2. Git and Repository Integrity

- **Active Prototype Repository:** `C:\Wira\Kuliah\Penelitian\PROTOTYPE APLIKASI FL\centralized-server-xai-multimodal-depression-fl`
- **Branch:** `main` (up to date with `origin/main`)
- **Git Commit Baseline:** `828c079d5c4ecffb935d77d93e9d48a9f683d612`
- **Working Tree Inspection:**
  - Untracked files present: `federated/` architecture, `configs/`, `storage/federated/`, `scripts/`, `docs/`, `tests/`
  - Zero unstaged deletions or corruptions to core production application (`app/core/`, `app/api/`, `app/db/`).

---

## 3. Test Suite Execution

Executed comprehensive test suite with 40 automated tests:
```
platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.6.0
================== 40 passed, 1 warning in 64.51s ==================
```
- **FedAvg Mathematical Aggregation:** PASSED (2 tests)
- **Checkpointing, Communication & Resume Lifecycle:** PASSED (7 tests)
- **Flower ClientApp, ArrayRecord & Privacy Safety:** PASSED (3 tests)
- **Dataset Partitioning & Zero Leakage:** PASSED (4 tests)
- **Label Skew Verification (M5.2 unit & 3-round smoke):** PASSED (12 tests)
- **FL Integration Smoke Test:** PASSED (2 tests)
- **Application Health, Model Info & P600 Golden Regression:** PASSED (10 tests)

---

## 4. Federated Architecture & Environment Verification

- **Flower Framework Version:** `flwr==1.40.0`
- **PyTorch Version:** `2.11.0+cu128` (CUDA available: `True`, Device: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`, CUDA 12.8, cuDNN 91900)
- **Python Version:** `3.14.3` (tags/v3.14.3:323c59a)
- **Architecture Model:** `MultimodalBaselineModel` (7,832,001 parameters)
- **ServerApp / ClientApp Communication:** Modern Flower 1.40 Message API (`Grid`, `Context`, `Message`, `ArrayRecord`, `MetricRecord`, `ConfigRecord`)

---

## 5. M5.1 Control Baseline Verification & Cryptographic Hashes

- **Official FL-IID Control Run Directory:**
  `storage/federated/runs/FL-RUN-20261016-130403-45D587/`
- **Round-000 Initial Model Path:**
  `storage/federated/runs/FL-RUN-20261016-130403-45D587/rounds/round_000/model.pt`
- **Round-000 SHA-256 Checksum:**
  `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973`
  - Verification: **EXACT MATCH** (integrity confirmed)
- **Total Checkpoints in M5.1:** Exactly 51 checkpoints (`round_000` through `round_050`)
- **Status in Manifest:** `COMPLETED`, 50 rounds executed, 3 participating nodes.

### Dataset File Checksums & Integrity
| Dataset Role | File Path | Samples | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **Global Test** | `clinical-xai-multimodal-depression-fl/federated_learning_output/global_test_index.csv` | 56 | `27ea4f24fb8aa32fcbc79d06ef7d833c74de1abfc5a7fbd1f76a3d7b07884984` |
| **Shared Validation** | `clinical-xai-multimodal-depression-fl/federated_learning_output/IID_3Clients/client_1_val.csv` | 56 | `54a1dd5c7ea1a5449df2f4e63a623858765662e5d588699b9f4efa3a6a6728ad` |
| **IID Client 1 Train** | `clinical-xai-multimodal-depression-fl/federated_learning_output/IID_3Clients/client_1_train.csv` | 54 | `e0b4d1282936b5a252a9a2c1ccecf2285a97ca00506e7e8cf2089b3d4f286524` |
| **IID Client 2 Train** | `clinical-xai-multimodal-depression-fl/federated_learning_output/IID_3Clients/client_2_train.csv` | 55 | `76d72f0480d90163ec540882b8971ffd26d0c1182b76b3a2170ecd537c101f09` |
| **IID Client 3 Train** | `clinical-xai-multimodal-depression-fl/federated_learning_output/IID_3Clients/client_3_train.csv` | 54 | `ff5cb8e3501113609b15b484719215deebe970d0337ca4bd2d4bf0b4da396355` |
| **Total IID Training Pool** | Union(Client 1, 2, 3) | 163 | 163 unique participant IDs, zero duplicates, zero test leakage |

---

## 6. M5.1 Reference Performance Baseline (Round 50)

Extracted directly from `storage/federated/runs/FL-RUN-20261016-130403-45D587/manifest.json`:
- **MAE:** 4.9997
- **RMSE:** 6.1402
- **MSE:** 37.7017
- **Macro F1:** 0.7005
- **Binary F1:** 0.6364
- **Recall / Sensitivity:** 0.6667
- **Specificity:** 0.7429
- **ROC-AUC:** 0.7592
- **Accuracy:** 0.7143
- **Confusion Matrix:** TP=14, FP=9, TN=26, FN=7 (Total=56)
- **FNR:** 0.3333 (7 / 21)
- **FPR:** 0.2571 (9 / 35)

---

## 7. Isolation Assessment of M5.2 Code Changes

- Milestone 5.2 (Label Skew) introduced experimental partition generation (`federated/partition_generator.py` and `scripts/design_label_skew.py`) and test suites (`tests/test_federated_label_skew.py`).
- The core federated engine (`federated/server_app.py`, `federated/client_app.py`, `federated/data.py`, `federated/checkpointing.py`) uses scenario parameters and did not modify the model architecture, loss function, or evaluation logic.
- Production models in `storage/models/centralized/` and `storage/models/fl_iid/` remain strictly unmodified.
- Milestone 5.3 will use M5.1 exclusively as its control baseline.

---

## 8. Conclusion and Pre-Flight Gate Verdict

**PRE-FLIGHT VERIFICATION GATE: PASSED**
Integrity of M5.1 control baseline, dataset cohort hashes, initial checkpoint SHA-256, and test suite is 100% verified. Proceeding to historical audit and partition design.
