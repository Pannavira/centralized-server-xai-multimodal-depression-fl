# FED-MIND — Milestone 5.5 Preflight Verification Report
**Controlled Combined Non-IID Federated Learning Pre-Execution Audit**  
**Date:** 2026-10-19  
**Status:** ALL PREFLIGHT GATES PASSED (READY FOR IMPLEMENTATION)

---

## 1. Executive Summary

Prior to implementing, executing, and evaluating **Milestone 5.5 (Controlled Combined Non-IID)**, a rigorous preflight audit was conducted across baseline control artifacts, previous milestone runs (M5.1, M5.2, M5.3, M5.4), repository status, software environments, and production model checkpoints.

Milestone 5.5 integrates three orthogonal heterogeneity mechanisms simultaneously:
1. **Label Skew:** Diverse depression-positive prevalence across clients.
2. **Quantity Skew:** Asymmetric sample counts (98 / 49 / 16, target ratio ~60% / 30% / 10%).
3. **Modality Skew:** Sensor quality degradation (Client 1: Clean; Client 2: Audio Gaussian noise $\sigma=0.25$; Client 3: Visual frame dropout $20\%$; Text: 100% clean across all clients).

All preflight checks have passed without exception:
- **Baseline Round-0 Checksum:** Bitwise identical to M5.1 (`a1c05247...`).
- **All 4 Prior Milestones Verified:** M5.1, M5.2, M5.3, and M5.4 artifacts present with verifiable metrics.
- **Production Models:** 100% untouched and protected.
- **Participant Pool:** 163 unique training participants strictly preserved.
- **Evaluation Cohorts:** Global test index ($N=56$) and shared validation ($N=56$) verified with exact SHA-256 hashes and zero leakage.
- **Automated Test Suite:** 85/85 tests passing.

---

## 2. Environment and Dependency Verification

| Component | Verified Specification | Audit Status |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 (10.0.26200-SP0) | VERIFIED |
| **Python Runtime** | Python 3.14.3 (tags/v3.14.3:323c59a) | VERIFIED |
| **PyTorch Version** | 2.14.1+cpu | VERIFIED |
| **Flower Framework** | Flower 1.40.0 (Message API: ServerApp, ClientApp, ArrayRecord) | VERIFIED |
| **Ray Version** | 2.59.0 | VERIFIED |
| **CUDA Acceleration** | None (CPU Simulation) | VERIFIED |
| **Active Prototype Repo** | `C:\Wira\Kuliah\Penelitian\PROTOTYPE APLIKASI FL\centralized-server-xai-multimodal-depression-fl` | VERIFIED |
| **Research Reference Repo** | `C:\Wira\Kuliah\Penelitian\clinical-xai-multimodal-depression-fl` | VERIFIED |
| **Git Working Tree** | Clean tracked tree; M5.4 artifacts staged for commit | VERIFIED |
| **Automated Test Suite** | 85 passed, 0 failed in 69.69s | VERIFIED |

---

## 3. Checkpoint and Cohort Integrity

### 3.1 Baseline Control Checkpoints (M5.1 Control)
- **Target File:** `storage/federated/runs/FL-RUN-20261016-130403-45D587/rounds/round_000/model.pt`
- **Expected SHA-256:** `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973`
- **Computed SHA-256:** `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973`
- **Integrity Status:** **PASSED (BITWISE IDENTICAL)**
- **Strict Architecture State Dict Loading:** Verified on `MultimodalBaselineModel` (7,832,001 parameters).

### 3.2 Evaluation Cohort Hashes
- **Global Hold-Out Test:** `federated_learning_output/global_test_index.csv`
  - SHA-256: `27ea4f24fb8aa32fcbc79d06ef7d833c74de1abfc5a7fbd1f76a3d7b07884984` ($N=56$, 21 positive, 35 negative)
- **Shared Validation Cohort:** `federated_learning_output/IID_3Clients/client_1_val.csv`
  - SHA-256: `54a1dd5c7ea1a5449df2f4e63a623858765662e5d588699b9f4efa3a6a6728ad` ($N=56$, 21 positive, 35 negative)
- **Zero-Leakage Assertion:** Confirmed train $\cap$ test $= \emptyset$, train $\cap$ val $= \emptyset$, val $\cap$ test $= \emptyset$.

### 3.3 Visual Imputation Reference Artifacts
- **Primary Training Mean Visual Frame:** `storage/models/train_visual_mean_frame.pt`
  - SHA-256: `19520e7fee2dba83119b8b85202acac62c96c93adfad92e1fd9d6de28ff707eb` (Shape: `[72, 3]`, mean scalar $= 0.227831$)
- **Legacy Visual Mean Frame:** `storage/models/visual_mean_frame.pt`
  - SHA-256: `d496c38fdd9f0766b3a423232fb6368b6aba43bfbaaacd7a085c2b2d0b465546`

### 3.4 Production Model Protection
All production screening models remain strictly protected and unmodified:
- `storage/models/fl_iid/model.pt`: `17d59aa90455386a1e2f00bed2107ef0cbe2a6fb3e96e3f476faf0272db0ec73`
- `storage/models/fl_non_iid/model.pt`: `8f4c574133c8a86c35d3525a65cc3d6b5ffdb302ce00ad670955dcb313b8bd0c`
- `storage/models/centralized/model.pt`: `df9168fe878f6de1c15d2e05b109302fbca49087dd89108be76ead24581c0939`

---

## 4. Prior Milestone Baseline Summary

All four prior milestones have been audited from their official stored runs:

| Milestone | Run ID | Scenario | MAE ↓ | RMSE ↓ | Macro F1 ↑ | Recall ↑ | Spec ↑ | AUC ↑ | FNR ↓ | FPR ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M5.1 Control** | `FL-RUN-20261016-130403-45D587` | `IID_3Clients` | 4.9997 | 6.1402 | 0.7005 | 0.6667 | 0.7429 | 0.7592 | 0.3333 | 0.2571 |
| **M5.2 Label Skew** | `FL-LABEL-SKEW-20261016-141108-7865D4` | `non_iid_label_skew` | 4.6964 | 5.9384 | 0.6725 | 0.4762 | 0.8571 | 0.7116 | 0.5238 | 0.1429 |
| **M5.3 Quantity Skew** | `FL-QUANTITY-SKEW-20261019-110746-3A4BFF` | `non_iid_quantity_skew` | 5.6170 | 6.8015 | 0.5892 | 0.8095 | 0.4571 | 0.7293 | 0.1905 | 0.5429 |
| **M5.4 Modality Skew** | `FL-MODALITY-SKEW-20261019-142242-CA6D11` | `non_iid_modality_skew` | 4.9739 | 6.0214 | 0.7214 | 0.7143 | 0.7429 | 0.7741 | 0.2857 | 0.2571 |

---

## 5. Architectural & Code Modifications Required for M5.5

1. **Partition Generator (`federated/partitions/combined_skew.py`):**
   - Create deterministic generator allocating 98 / 49 / 16 sample counts with label stratification:
     - Client 1: 18 positive, 80 negative ($18.37\%$ prevalence)
     - Client 2: 19 positive, 30 negative ($38.78\%$ prevalence)
     - Client 3: 13 positive, 3 negative ($81.25\%$ prevalence)
   - Store in `storage/federated/partitions/combined_skew_seed42/`.
2. **Scenario Routing (`federated/data.py` & `federated/server_app.py` & `federated/client_app.py`):**
   - Add support for scenario key `non_iid_combined` / `controlled_combined_skew_v1`.
   - Route `degradation_config` to activate Client 1 (Clean), Client 2 (Audio noisy $\sigma=0.25$), Client 3 (Visual frame dropout $0.20$ with neutral mean imputation).
3. **Automated Testing Suite (`tests/test_federated_combined_skew.py`):**
   - Assert all 30 scientific invariants (pool equality, sample quantities, label prevalence, zero leakage, degradation manifestation, etc.).
4. **Comparison Module (`federated/comparison.py`):**
   - Implement `generate_combined_skew_comparison` to compare M5.5 against M5.1, M5.2, M5.3, and M5.4.
5. **Configuration (`pyproject.toml`):**
   - Set scenario to `non_iid_combined`, scientific-status to `controlled_experiment_combined_skew`.

---

## 6. Known Scientific Limitations

- **Confounded Factors:** With only 3 clients, Client 1 is simultaneously large, negative-majority, and clean; Client 2 is medium, mixed-prevalence, and audio-noisy; Client 3 is small, positive-majority, and visually degraded. The experiment observes global FedAvg behavior under joint stress, but cannot isolate independent interaction terms.
- **Single Random Seed ($42$):** Results reflect observed performance at seed 42. Statistical confidence intervals across seeds will be addressed in Milestone 5.6.
