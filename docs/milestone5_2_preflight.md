# Milestone 5.2 Pre-Flight Freeze: M5.1 FL-IID Control Experiment

**Document**: `docs/milestone5_2_preflight.md`  
**Milestone**: Phase 4 — Milestone 5.2 (Controlled Non-IID Label Skew Experiment)  
**Execution Timestamp**: 2026-10-16  
**Status**: **FROZEN & VERIFIED (PRE-FLIGHT PASSED)**  

---

## 1. M5.1 Control Run Identity & Integrity Verification

- **Official M5.1 Control Run ID**: `FL-RUN-20261016-130403-45D587`
- **Run Directory**: `storage/federated/runs/FL-RUN-20261016-130403-45D587/`
- **Checkpoint Range**: `round_000` through `round_050` (51 complete checkpoints verified)
- **Round 0 Checkpoint Path**: `storage/federated/runs/FL-RUN-20261016-130403-45D587/rounds/round_000/model.pt`
- **Round 0 SHA-256 Checksum**:
  `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973`
- **Strict Architecture Compatibility**:
  Checkpoint `round_000/model.pt` loaded successfully into authoritative `MultimodalBaselineModel` with `strict=True` (79 parameters, 0 missing keys, 0 unexpected keys).

> [!IMPORTANT]
> **Pre-Flight Invariant Assertion**: Checksum Round 0 matches `a1c052476e...` exactly. M5.2 will branch deterministically from this exact same checkpoint initialization.

---

## 2. Frozen M5.1 Configuration Parameters

| Parameter | Frozen Value (M5.1 Control) | Requirement for M5.2 |
|---|---|---|
| **Federated Framework** | Flower 1.40.0 (`ServerApp`, `ClientApp`, Message API) | Identical |
| **Federated Algorithm** | FedAvg (`weighted_by="num-examples"`) | Identical |
| **Number of Clients** | 3 Clients | Identical |
| **Client Participation** | 100% (3 / 3 active every round) | Identical |
| **Number of Rounds** | 50 FedAvg Communication Rounds | Identical |
| **Local Epochs per Round** | 2 Epochs | Identical |
| **Batch Size** | 16 | Identical |
| **Optimizer** | AdamW | Identical |
| **Client Learning Rate** | $1 \times 10^{-4}$ (0.0001) | Identical |
| **Weight Decay** | 0.01 | Identical |
| **Learning Rate Scheduler** | `CosineAnnealingLR` ($T_{\max}=\text{local\_epochs}$, reset per local round) | Identical |
| **Loss Function** | `ClassWeightedMSELoss` ($w_{\text{pos}}=2.5$, penalty $FN=2.0$) | Identical |
| **Clinical PHQ-8 Cutoff** | $\ge 10.0$ | Identical |
| **Random Seed** | 42 | Identical |
| **Model Selection Policy** | Hold-out Global Test reporting only (No test-based selection) | Identical |
| **Official Evaluated Model** | Final Round 50 Model | Identical |

---

## 3. Frozen M5.1 Control Baseline Metrics

### A. Round 0 (Initial Baseline)
- **Test Loss**: `196.7155`
- **MAE**: `8.1071`
- **RMSE**: `10.3077`
- **Macro F1**: `0.3846`
- **Recall**: `0.0000`
- **Specificity**: `1.0000`
- **ROC-AUC**: `0.4313`

### B. Round 50 (Official Control Baseline)
- **Test Loss**: `42.6409`
- **MAE**: `4.9997`
- **RMSE**: `6.1402`
- **MSE**: `37.7017`
- **Macro F1**: `0.7005`
- **Recall / Sensitivity**: `0.6667`
- **Specificity**: `0.7429`
- **ROC-AUC**: `0.7592`
- **Lin's CCC**: `0.4345`
- **Pearson $r$**: `0.4524`
- **Confusion Matrix**:
  - True Positives (TP): `14`
  - False Negatives (FN): `7`
  - True Negatives (TN): `26`
  - False Positives (FP): `9`
  - False Negative Rate (FNR = FN / (TP + FN)): `0.3333` (33.3%)
  - False Positive Rate (FPR = FP / (FP + TN)): `0.2571` (25.7%)

---

## 4. Frozen M5.1 Communication Accounting

- **Parameters per Checkpoint**: `7,832,001` float32 ($\approx 29.8767$ MB)
- **Downlink Payload per Round**: $3 \times 29.8767\text{ MB} = 89.6301\text{ MB}$
- **Uplink Payload per Round**: $3 \times 29.8767\text{ MB} = 89.6301\text{ MB}$
- **Total Payload per Round**: $179.2603\text{ MB}$
- **Cumulative 50-Round Payload**: $9,398,401,200\text{ bytes} \approx 8.7529\text{ GB}$

---

## 5. Pre-Flight Conclusion
M5.1 is officially frozen. All invariants and baseline figures are logged and locked.
The system is clear to proceed to **Historical Label-Skew Audit (Step 2)** and **Controlled Partition Design (Steps 4–6)**.
