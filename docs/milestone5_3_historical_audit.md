# FED-MIND — Milestone 5.3: Historical Quantity-Skew Audit Report
**Analysis of Historical Artifacts vs. Controlled Scientific Requirements**
**Date:** 2026-10-19  
**Status:** COMPLETED & AUDITED

---

## 1. Executive Summary

This audit examines the historical Quantity Skew partition files, scripts, and configurations in the reference research repository (`C:\Wira\Kuliah\Penelitian\clinical-xai-multimodal-depression-fl`).

**Key Scientific Verdict:**  
The historical Quantity Skew partition located at `federated_learning_output/NonIID_QuantitySkew` **CANNOT and MUST NOT be directly reused** for Milestone 5.3. Doing so would violate experimental controls: only 92 participants overlap with the M5.1 pool, meaning **71 out of 163 M5.1 participants were omitted** and **56 extraneous participants outside the M5.1 pool were introduced** (with a net sample size difference of 15: 148 vs 163). Furthermore, it leaked/fragmented the validation cohort (14/9/5 local split vs shared 56) and used an approximate 10:6:3 ratio instead of the protocol-mandated 60%:30%:10% ratio. A new deterministic, stratified partition (`storage/federated/partitions/quantity_skew_seed42`) must be generated to preserve the exact M5.1 participant pool (`S_IID == S_QUANTITY`) and ensure a rigorous ceteris paribus comparison.

---

## 2. Historical Research Artifacts Inspection

### 2.1 File Locations and Properties
The historical research repository contains:
- Directory: `C:\Wira\Kuliah\Penelitian\clinical-xai-multimodal-depression-fl\federated_learning_output\NonIID_QuantitySkew`
  - `client_1_train.csv`: 79 samples (7,463 bytes)
  - `client_2_train.csv`: 46 samples (4,381 bytes)
  - `client_3_train.csv`: 23 samples (2,237 bytes)
  - **Total Training Samples:** 148
  - `client_1_val.csv`: 14 samples
  - `client_2_val.csv`: 9 samples
  - `client_3_val.csv`: 5 samples
  - **Total Validation Samples:** 28
  - `quantity_skew_distribution_report.xlsx`
  - `quantity_skew_partition_log.xlsx`

### 2.2 Historical Partitioning Logic
In `federated_learning/script_non_iid_partition.py` (`make_quantity_skew`):
```python
def make_quantity_skew(df_train, num_clients, random_state):
    # Ratios targeted: 10:6:3 (~52.6% / 31.6% / 15.8%)
    # Hardcoded candidate sizes: 93, 55, 28 from a 176-participant pool
    # Followed by a local random validation split (val_size=0.15)
    # Resulting client train sizes: 79, 46, 23 (Total = 148)
```

---

## 3. Discrepancy & Confounding Analysis

A rigorous comparison between the M5.1 IID Control cohort and the Historical Quantity Skew cohort reveals major scientific discrepancies:

| Attribute | M5.1 IID Control | Historical NonIID_QuantitySkew | Status / Impact |
| :--- | :--- | :--- | :--- |
| **Total Training Participants** | 163 | 148 | **Net size delta of 15 (148 vs 163)** |
| **Client 1 Training Size** | 54 (33.13%) | 79 (53.38%) | Ratio ~ 10:6:3 (~53.4% : 31.1% : 15.5%) |
| **Client 2 Training Size** | 55 (33.74%) | 46 (31.08%) | Ratio ~ 10:6:3 (~53.4% : 31.1% : 15.5%) |
| **Client 3 Training Size** | 54 (33.13%) | 23 (15.54%) | Ratio ~ 10:6:3 (~53.4% : 31.1% : 15.5%) |
| **Validation Cohort** | Shared 56 participants | Separate per-client: 14 / 9 / 5 | **Validation fragmentation / leakage** |
| **Global Test Set** | Fixed 56 participants | Fixed 56 participants | Preserved |
| **Participant Pool Overlap** | Reference: 163 IDs | Intersection: 92 IDs ($S_{\text{HIST}} \cap S_{\text{M5.1}}$) | **71 of 163 M5.1 IDs missing (43.6%); 56 extraneous IDs added** |
| **Invariant $S_{IID} == S_{QUANTITY}$** | N/A | **VIOLATED** | Severe cohort confounding |

### Why Historical Partition Cannot Be Reused:
1. **Participant Confounding:** Reusing the 148-sample historical partition means training on a substantially different cohort. With 71 M5.1 participants missing and 56 extraneous participants added (only 92 in common), any observed divergence from IID would stem from sample composition bias rather than pure quantity skew. The net count difference of 15 (163 vs 148) obscures the fact that nearly half of the M5.1 cohort was absent.
2. **Validation Inconsistency:** The historical partition fragmented validation into 14, 9, and 5 samples, making fair round-by-round client validation comparisons impossible. M5.1 uses a standardized shared validation cohort (56 participants).
3. **Ratio Deviation:** The research protocol specifically mandates the canonical 60% : 30% : 10% distribution (98 / 49 / 16 for 163 participants), whereas the historical partition used ~53% : 31% : 16%.

---

## 4. Parameter Distinction Matrix

| Dimension | Historically Verified Setting | New Controlled M5.3 Design Choice | Unknown / Uncontrolled Historical Setting |
| :--- | :--- | :--- | :--- |
| **Clients** | 3 clients | 3 clients | — |
| **Target Distribution** | Approximate 10:6:3 | Exact integer **98 / 49 / 16** (60.12% / 30.06% / 9.82%) | Ad-hoc rounding |
| **Training Population** | 148 participants | Exactly **163 participants** ($S_{QUANTITY} \equiv S_{IID}$) | Patient inclusion criteria |
| **Validation Cohort** | Local split (14/9/5) | **Standardized shared validation cohort (56)** | Local val leakage |
| **Global Test Cohort** | Fixed 56 hold-out | **Identical fixed 56 hold-out** | — |
| **Label Preservation** | Unconstrained | **Deterministic stratified allocation** (Seed 42) minimizing label deviation | Residual label bias |
| **Model Architecture** | MultimodalBaselineModel | **Identical MultimodalBaselineModel** | — |
| **Initial Checkpoint** | Random / Pretrained | **M5.1 Round 000 Checkpoint** (`a1c05247...`) | Pretrained weights |
| **Federated Aggregation** | FedAvg | **FedAvg (`weighted_by_key="num-examples"`)** | Default aggregation |
| **Hyperparameters** | AdamW, lr=1e-4, 50 rounds | **Identical M5.1 configuration** | Dynamic adjustments |

---

## 5. Architectural Decision

1. **Independent Partition Generation:** Implement `federated/partitions/quantity_skew.py` to deterministically allocate the official 163 M5.1 participants into capacities `[98, 49, 16]`.
2. **Deterministic Stratification:** Ensure that the 50 positive (PHQ-8 $\ge$ 10) and 113 negative (PHQ-8 $<$ 10) participants are partitioned so that each client's positive prevalence is as close as possible to the global training prevalence ($50 / 163 \approx 30.67\%$).
3. **Storage Location:** Save new partition artifacts in `storage/federated/partitions/quantity_skew_seed42/`. Do not modify or overwrite historical files in `clinical-xai-multimodal-depression-fl`.
