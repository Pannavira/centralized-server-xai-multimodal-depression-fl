# FED-MIND — Milestone 5.5 Historical Combined Non-IID Forensic Audit
**Forensic Audit of Historical Partitions and Experiment Artifacts in Research Repository**  
**Date:** 2026-10-19  
**Status:** AUDIT COMPLETED — HISTORICAL PARTITIONS REJECTED DUE TO DATA LEAKAGE & COHORT DRIFT

---

## 1. Executive Summary

A forensic audit of historical artifacts in the reference research repository (`C:\Wira\Kuliah\Penelitian\clinical-xai-multimodal-depression-fl`) was conducted to evaluate whether historical Combined Non-IID partitions could be reused or referenced for Milestone 5.5.

**Critical Finding:**  
Historical Combined Non-IID partitions (`federated_learning_output/NonIID_Combined/`) and historical experiment outputs (`experiments_v2/fedavg_noniid_combined_pretrained/`) are **scientifically invalid for a controlled FL experiment**. They exhibit:
1. **Severe Data Leakage:** 31 subjects from `global_test_index.csv` ($55.4\%$ of the test set) leaked into client training sets.
2. **Cohort Incompatibility:** Total training pool was 148 subjects ($N_{\text{hist}} = 148 \ne 163$); 72 official M5.1 training subjects were missing, and 57 external subjects were introduced.
3. **Hyperparameter & Initialization Incompatibility:** Pretrained initialization from `modeling2/output/best_model.pt` was used instead of Round-0 weights, with 20 rounds, different learning rate, and auxiliary loss terms.

Consequently, historical partitions cannot be reused. A **NEW CONTROLLED DESIGN (`controlled_combined_skew_v1`)** must be established directly from the verified 163-subject M5.1 cohort.

---

## 2. Forensic Audit Findings

| Audit Dimension | Historical Artifact Finding | Status | Evaluation for M5.5 Control |
| :--- | :--- | :--- | :--- |
| **Directory Location** | `clinical-xai-multimodal-depression-fl/federated_learning_output/NonIID_Combined/` | HISTORICALLY VERIFIED | Found on disk |
| **Partition Sample Counts** | Client 1: 79, Client 2: 46, Client 3: 23 (Total = 148) | HISTORICALLY VERIFIED | REJECTED ($148 \ne 163$) |
| **Global Test Leakage** | 31 subjects overlap with `global_test_index.csv` | HISTORICALLY VERIFIED | CRITICAL VIOLATION (Zero Leakage Broken) |
| **Cohort Identity** | Pool $\ne S_{\text{M5.1}}$ (72 missing, 57 added) | HISTORICALLY VERIFIED | REJECTED (Cohort Drift) |
| **Label Allocation** | C1: 32 pos (40.5%), C2: 13 pos (28.3%), C3: 2 pos (8.7%) | HISTORICALLY VERIFIED | REJECTED (Contaminated) |
| **Model Initialization** | `modeling2/output/best_model.pt` (Centralized Pretrained) | HISTORICALLY VERIFIED | REJECTED (Must start from R0) |
| **Server Rounds** | 20 Communication Rounds | HISTORICALLY VERIFIED | REJECTED (M5.1 uses 50 rounds) |
| **Learning Rate** | $5 \times 10^{-5}$ | HISTORICALLY VERIFIED | REJECTED (M5.1 uses $1 \times 10^{-4}$) |
| **Loss Function** | MSE + 0.2 CCC + 0.2 BCE | HISTORICALLY VERIFIED | REJECTED (M5.1 uses ClassWeightedMSE) |
| **Reported Test Metrics** | MAE: 3.8822, RMSE: 4.9346, Macro F1: 0.8085 (Round 15) | HISTORICALLY VERIFIED | ARTIFICIAL (Driven by 31 test subjects in train) |

---

## 3. Detailed Data Leakage Manifest

The 31 leaked subjects present in historical `NonIID_Combined` client training files:
```text
Leaked Participant IDs (N = 31):
600, 602, 605, 607, 609, 619, 625, 629, 634, 635, 636, 637, 638, 640,
651, 658, 663, 664, 666, 669, 676, 682, 683, 689, 693, 696, 699, 705,
710, 716, 717
```
Because more than half of the 56 evaluation subjects were seen during local training, historical test metrics are heavily inflated and cannot serve as legitimate scientific evidence.

---

## 4. Controlled Design Protocol (`controlled_combined_skew_v1`)

To maintain the ceteris paribus invariant against M5.1 Control, M5.2 Label Skew, M5.3 Quantity Skew, and M5.4 Modality Skew:

1. **Cohort Source:** Exactly the 163 training participants from M5.1 ($S_{\text{COMBINED}} \equiv S_{\text{M5.1}}$).
2. **Quantity Distribution:**
   - Client 1: 98 participants ($60.12\%$, FedAvg weight $98/163$)
   - Client 2: 49 participants ($30.06\%$, FedAvg weight $49/163$)
   - Client 3: 16 participants ($9.82\%$, FedAvg weight $16/163$)
3. **Controlled Label Allocation:**
   - Client 1: 18 positive, 80 negative ($18.37\%$ prevalence — Negative-Heavy)
   - Client 2: 19 positive, 30 negative ($38.78\%$ prevalence — Mixed)
   - Client 3: 13 positive, 3 negative ($81.25\%$ prevalence — Positive-Heavy)
   - Global: 50 positive, 113 negative ($30.67\%$ prevalence)
4. **Modality Quality Assignment:**
   - Client 1: Clean (Audio, Visual, Text)
   - Client 2: Audio Gaussian noise ($\sigma = 0.25$), Clean Visual, Clean Text
   - Client 3: Visual frame dropout ($20\%$, neutral mean imputation), Clean Audio, Clean Text
5. **Deterministic Seed:** Seed 42 for all assignment and corruption logic.
6. **Zero Leakage:** Complete disjointness with global test ($N=56$) and shared validation ($N=56$).
