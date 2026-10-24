# FED-MIND — Milestone 5.4 Preflight Verification Report
**Controlled Non-IID Modality Skew Pre-Execution Audit**  
**Date:** 2026-10-19  
**Status:** ALL PREFLIGHT GATES PASSED (READY FOR IMPLEMENTATION)  

---

## 1. Executive Summary

Prior to implementing and executing Milestone 5.4 (Controlled Non-IID Modality Skew), a comprehensive preflight validation was performed across baseline checkpoints, model architectures, dataset pipelines, tensor contracts, and normalization statistics.

All preflight criteria have passed without exceptions:
- **M5.1 Round-0 SHA-256 Checksum:** MATCHED (`a1c05247...`)
- **Strict Architecture State Dict Loading:** PASSED (`MultimodalBaselineModel`, 7,832,001 parameters)
- **M5.1 Round-50 Benchmark Verification:** VERIFIED directly from stored artifacts
- **Zero-Leakage & Participant Cohort:** VERIFIED ($S_{\text{M5.4}} \equiv S_{\text{M5.1}}$, 163 training participants, 0 test overlap)
- **Multimodal Tensor Contract:** VERIFIED on active sample tensors

---

## 2. Checkpoint and Initialization Verification

### 2.1 Round 0 Checkpoint Verification
- **Target File:** `storage/federated/runs/FL-RUN-20261016-130403-45D587/rounds/round_000/model.pt`
- **Expected SHA-256:** `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973`
- **Computed SHA-256:** `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973`
- **Integrity Status:** **PASSED (BITWISE IDENTICAL)**
- **Strict Loading Validation:** `MultimodalBaselineModel.load_state_dict(sd, strict=True)` loaded without missing or unexpected keys.

### 2.2 Round 50 Primary Control Metrics (M5.1 FL-IID)
Metrics extracted directly from `storage/federated/runs/FL-RUN-20261016-130403-45D587/metrics.csv` and confirmed via independent centralized evaluation (`GlobalEvaluator`):

| Metric | Stored M5.1 Value | Re-Evaluated M5.1 Value | Verification Status |
| :--- | :--- | :--- | :--- |
| **MAE** | 4.9997 | 4.9997 | MATCHED |
| **RMSE** | 6.1402 | 6.1402 | MATCHED |
| **MSE** | 37.7017 | 37.7017 | MATCHED |
| **Macro F1** | 0.7005 | 0.7005 | MATCHED |
| **Binary F1** | 0.6364 | 0.6364 | MATCHED |
| **Recall / Sensitivity** | 0.6667 | 0.6667 | MATCHED |
| **Specificity** | 0.7429 | 0.7429 | MATCHED |
| **ROC-AUC** | 0.7592 | 0.7592 | MATCHED |
| **Lin's CCC** | 0.4345 | 0.4345 | MATCHED |
| **Pearson r** | 0.4524 | 0.4524 | MATCHED |
| **TP / FP / TN / FN** | 14 / 9 / 26 / 7 | 14 / 9 / 26 / 7 | MATCHED |
| **FNR** | 0.3333 (7/21) | 0.3333 (7/21) | MATCHED |
| **FPR** | 0.2571 (9/35) | 0.2571 (9/35) | MATCHED |
| **Test Loss** | 42.6409 | 42.6409 | MATCHED |
| **Total Test Subjects**| 56 | 56 | MATCHED |

---

## 3. Dataset Pipeline and Multimodal Tensor Contract

### 3.1 Verified Tensor Contract
Auditing active training samples across Client 1, Client 2, Client 3, and Global Hold-Out test sets establishes the exact operational contract:

| Modality | Raw Feature Shape | Adapted Tensor Shape | PyTorch Dtype | Preprocessing Pipeline | Padding / Temporal Semantics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Audio** | `(248,)` | `(512, 80)` | `torch.float32` | Tiled to 80 channels $\times$ temporal slope $[0.8, 1.2]$ | Full sequence active ($T=512$); padded with 0.0 if sequence length $< 512$ |
| **Visual** | `(75,)` | `(512, 72, 3)` | `torch.float32` | Tiled to 216 ($72 \times 3$), min-max scaled to $[0, 1] \times$ temporal slope $[0.9, 1.1]$ | Full sequence active ($T=512$); reshaped to 72 landmark nodes and 3 coordinates $(x, y, z)$ |
| **Text** | `(768,)` | `(512, 768)` | `torch.float32` | Tiled to 768 $\times$ temporal slope $[0.95, 1.05]$ | Full sequence active ($T=512$); padded with 0.0 if $< 768$ |
| **PHQ-8 Score** | scalar | `()` | `torch.float32` | Continuous regression ground truth $[0, 24]$ | Scalar tensor |
| **Depression Label** | scalar | `()` | `torch.int64` | Binary threshold $\ge 10.0$ | Scalar long tensor ($0$ or $1$) |

### 3.2 Preprocessing and Normalization Audit
- **Audio Statistics:** Empirical training distribution across all 163 participants has global mean $= 1.9359$, $\text{std} = 13.8373$.
- **Visual Statistics:** Empirical training distribution across all 163 participants has global mean $= 0.2278$. An empirical neutral reference frame $\bar{\mathbf{V}} \in \mathbb{R}^{72 \times 3}$ is derived with values in $[0.0029, 0.7568]$.
- **Text Statistics:** Empirical training distribution has global mean $= -0.0108$, $\text{std} \approx 0.07$. Text remains 100% clean across all clients.

---

## 4. Controlled Modality Treatment Assignment

To guarantee that M5.4 isolates the effect of client modality quality heterogeneity without confounding from participant reallocation:

| Client Partition | Sample Count | Modality Treatment | Corruption Details |
| :--- | :--- | :--- | :--- |
| **Client 1** | 54 (33.13%) | **Full Clean Modalities** | Audio: Clean, Visual: Clean, Text: Clean |
| **Client 2** | 55 (33.74%) | **Audio Perturbation** | Audio: Gaussian feature noise ($\sigma = 0.25$), Visual: Clean, Text: Clean |
| **Client 3** | 54 (33.13%) | **Visual Frame Degradation** | Audio: Clean, Visual: Temporal Frame Dropout (20% Mean-Imputation), Text: Clean |
| **Shared Validation** | 56 | **Clean Baseline** | Audio: Clean, Visual: Clean, Text: Clean |
| **Global Test Set** | 56 | **Clean Hold-Out** | Audio: Clean, Visual: Clean, Text: Clean |

---

## 5. Deterministic Corruption Specification

1. **Deterministic Per-Sample Derivation:**
   ```python
   hash_key = f"{experiment_seed}_{client_id}_{sample_id}_{modality}_{protocol_version}"
   sample_seed = int(hashlib.sha256(hash_key.encode("utf-8")).hexdigest()[:8], 16)
   rng = torch.Generator()
   rng.manual_seed(sample_seed)
   ```
2. **Isolation Guarantee:**
   - A dedicated `torch.Generator()` is instantiated locally per sample.
   - Global PyTorch RNG, NumPy RNG, and Python random states are **NEVER** modified.
   - DataLoader shuffling, model dropout, and optimizer stochastic behavior remain completely unaffected.
3. **Reproducibility Guarantee:**
   - Repeated calls to `__getitem__` on the same sample produce bitwise identical corrupted tensors across all rounds and epochs.
4. **Zero-Leakage & Storage Guarantee:**
   - Corrupted features are generated dynamically in-memory; no corrupted arrays are persisted to disk or transmitted across Flower.

---

## 6. Preflight Invariant Matrix

| Requirement | Expected | Verified | Status |
| :--- | :--- | :--- | :--- |
| Total Training Cohort | 163 | 163 | PASS |
| Client 1 Samples | 54 | 54 | PASS |
| Client 2 Samples | 55 | 55 | PASS |
| Client 3 Samples | 54 | 54 | PASS |
| Unique Training IDs | 163 | 163 | PASS |
| Shared Validation Cohort | 56 | 56 | PASS |
| Global Test Cohort | 56 | 56 | PASS |
| Train $\cap$ Test Overlap | 0 | 0 | PASS |
| Train $\cap$ Val Overlap | 0 | 0 | PASS |
| Round 0 Model SHA-256 | `a1c05247...` | `a1c05247...` | PASS |
| Round 50 MAE / F1 Baseline | 4.9997 / 0.7005 | 4.9997 / 0.7005 | PASS |
| Audio Tensor Shape | `(512, 80)` | `(512, 80)` | PASS |
| Visual Tensor Shape | `(512, 72, 3)` | `(512, 72, 3)` | PASS |
| Text Tensor Shape | `(512, 768)` | `(512, 768)` | PASS |

Preflight verification is complete. The system is certified to proceed to module implementation.
