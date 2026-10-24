# FED-MIND — Milestone 5.4: Historical Modality-Skew Audit Report
**Analysis of Historical Artifacts vs. Controlled Scientific Requirements**  
**Date:** 2026-10-19  
**Status:** COMPLETED & AUDITED  

---

## 1. Executive Summary

This audit examines historical Modality Skew partition files, scripts, and experimental configurations in the reference research repository (`C:\Wira\Kuliah\Penelitian\clinical-xai-multimodal-depression-fl`).

**Key Scientific Verdict:**  
The historical Modality Skew partition located at `federated_learning_output/NonIID_ModalitySkew` **CANNOT and MUST NOT be directly reused** for Milestone 5.4. Doing so would violate experimental controls and introduce severe cohort confounding:
1. **Participant Invariant Violation:** The historical partition contains only 149 participants (Client 1: 49, Client 2: 50, Client 3: 50), whereas the M5.1 IID baseline cohort contains 163 participants. Only 93 participants overlap with M5.1 ($|S_{\text{HIST}} \cap S_{\text{M5.1}}| = 93$); **70 M5.1 participants were omitted** and **56 extraneous participants outside the M5.1 pool were introduced**.
2. **Confounding Transformation Methodology:** The historical implementation in `script_non_iid_partition.py` (`make_modality_skew`) did **NOT** apply sensor noise, frame corruption, or feature masking. Instead, it sorted participants based on natural metadata (`valid_frame_ratio` for Client 3, and `session_duration_sec` for Client 2) and redistributed them into non-identical client subsets.
3. **Training Inconsistencies:** Historical experiments (`fedavg_noniid_modalityskew_pretrained`) executed only 20 rounds, used warm-start initialization from centralized `best_model.pt`, and used local fragmented validation sets (`client_X_val.csv` of 10 samples each) instead of the standardized 56 shared validation set.

To isolate the causal effect of **sensor quality heterogeneity** on FedAvg, M5.4 adopts a **Controlled Feature-Space Degradation Protocol** (`controlled_modality_skew_v1`) applied strictly to local client training while holding the exact 163 M5.1 IID client participant allocations, sample quantities (54 / 55 / 54), labels, initial weights, and global test sets invariant ($S_{\text{M5.4}} \equiv S_{\text{M5.1}}$).

---

## 2. Historical Research Artifacts Inspection

### 2.1 File Locations and Properties
The historical research repository contains:
- Directory: `C:\Wira\Kuliah\Penelitian\clinical-xai-multimodal-depression-fl\federated_learning_output\NonIID_ModalitySkew`
  - `client_1_train.csv`: 49 samples (4,664 bytes)
  - `client_2_train.csv`: 50 samples (4,756 bytes)
  - `client_3_train.csv`: 50 samples (4,754 bytes)
  - **Total Training Samples:** 149
  - `client_1_val.csv`: 10 samples (930 bytes)
  - `client_2_val.csv`: 10 samples (929 bytes)
  - `client_3_val.csv`: 10 samples (930 bytes)
  - **Total Validation Samples:** 30
  - `modality_skew_distribution_report.xlsx` (6,486 bytes)
  - `modality_skew_partition_log.xlsx` (17,942 bytes)

- Experiment Directories:
  - `federated_learning_output/experiments/fedavg_noniid_modalityskew_randominit/`
  - `federated_learning_output/experiments_v2/fedavg_noniid_modalityskew_pretrained/`
  - `federated_learning_output/experiment_v2/fedavg_noniid_modalityskew_pretrained/`

- Centralized Ablation Reference:
  - `modeling2/output/output02_Moderate_Regularization_Drop04/modality_ablation_summary.csv`
  - Assesses complete modality removal (`text_only`, `visual_only`, `audio_only`, `text_audio`, etc.) during centralized inference, not federated client sensor heterogeneity.

### 2.2 Historical Partitioning Logic
In `federated_learning/script_non_iid_partition.py` (`make_modality_skew`):
```python
def make_modality_skew(df_train, num_clients, random_state):
    """
    Membagi data dengan skew kualitas modalitas (Client 3 visual rendah, 
    Client 2 audio noisy/pendek, Client 1 audio/video baik).
    Jumlah sampel dan rasio kelas antar klien tetap seimbang (IID label).
    """
    depressed = df_train[df_train['binary_label'] == 1].copy()
    non_depressed = df_train[df_train['binary_label'] == 0].copy()
    
    # 1. Client 3 (Low Visual): Bottom visual valid_frame_ratio
    dep_sorted_vis = depressed.sort_values(by='valid_frame_ratio', ascending=True)
    d_c3 = dep_sorted_vis.iloc[0:14]
    dep_rem = dep_sorted_vis.iloc[14:]
    non_sorted_vis = non_depressed.sort_values(by='valid_frame_ratio', ascending=True)
    n_c3 = non_sorted_vis.iloc[0:45]
    non_rem = non_sorted_vis.iloc[45:]
    
    # 2. Client 2 (Audio noisy/pendek): Bottom session_duration_sec dari sisa data
    dep_sorted_aud = dep_rem.sort_values(by='session_duration_sec', ascending=True)
    d_c2 = dep_sorted_aud.iloc[0:14]
    d_c1 = dep_sorted_aud.iloc[14:]
    non_sorted_aud = non_rem.sort_values(by='session_duration_sec', ascending=True)
    n_c2 = non_sorted_aud.iloc[0:45]
    n_c1 = non_sorted_aud.iloc[45:]
    
    c1 = pd.concat([d_c1, n_c1]).sample(frac=1.0, random_state=random_state)
    c2 = pd.concat([d_c2, n_c2]).sample(frac=1.0, random_state=random_state)
    c3 = pd.concat([d_c3, n_c3]).sample(frac=1.0, random_state=random_state)
    return [c1, c2, c3]
```

---

## 3. Discrepancy & Confounding Analysis

| Attribute | M5.1 IID Control | Historical NonIID_ModalitySkew | Status / Impact |
| :--- | :--- | :--- | :--- |
| **Total Training Participants** | 163 | 149 | **Net size delta of 14 (149 vs 163)** |
| **Client 1 Training Size** | 54 (33.13%) | 49 (32.89%) | Sample redistribution |
| **Client 2 Training Size** | 55 (33.74%) | 50 (33.56%) | Sample redistribution |
| **Client 3 Training Size** | 54 (33.13%) | 50 (33.56%) | Sample redistribution |
| **Validation Cohort** | Shared 56 participants | Separate per-client: 10 / 10 / 10 | **Validation fragmentation / leakage** |
| **Global Test Set** | Fixed 56 participants | Fixed 56 participants | Preserved |
| **Participant Pool Overlap** | Reference: 163 IDs | Intersection: 93 IDs ($S_{\text{HIST}} \cap S_{\text{M5.1}}$) | **70 of 163 M5.1 IDs missing (42.9%); 56 extraneous IDs added** |
| **Invariant $S_{IID} == S_{MODALITY}$** | Reference | **VIOLATED** | Severe cohort confounding |
| **Degradation Method** | Uncorrupted Clean | Metadata sorting (`valid_frame_ratio`, `session_duration_sec`) | Does not test actual sensor perturbation |
| **Initialization** | Round 0 seed 42 (`a1c05247...`) | Pretrained warm-start (`best_model.pt`) | Initialization disparity |
| **Communication Rounds** | 50 rounds | 20 rounds | Protocol disparity |

### Scientific Verdict on Historical Protocol Options:
- **A. Raw audio waveform noise:** Not applied in historical repository (features are pre-extracted).
- **B. Extracted audio feature noise:** Not applied in historical repository.
- **C. Visual landmark degradation:** Not applied in historical repository.
- **D. Visual frame dropout:** Not applied in historical repository.
- **E. Entire modality removal:** Tested only in centralized `modeling2` ablation (`modality_ablation_summary.csv`), not in FL training.
- **F. Another documented transformation:** **HISTORICALLY VERIFIED**. The historical repository used metadata sorting and participant reallocation based on `valid_frame_ratio` and `session_duration_sec`.

**Why the Historical Approach Must Be Rejected for M5.4:**
Reallocating participants based on natural metadata changes the patient cohort composition on each client, confounding modality quality with individual clinical differences (70 patients missing, 56 new patients introduced). To achieve true experimental control (*ceteris paribus*), the patient assignment must be identical to M5.1, and the modality degradation must be applied as a synthetic feature-space transformation during local training.

---

## 4. Parameter Distinction Matrix

| Dimension | Historically Verified Setting | New Controlled M5.4 Design Choice | Unknown / Uncontrolled Historical Setting |
| :--- | :--- | :--- | :--- |
| **Clients** | 3 clients | 3 clients | — |
| **Training Population** | 149 participants | Exactly **163 participants** ($S_{\text{MODALITY}} \equiv S_{\text{IID}}$) | Patient selection bias |
| **Per-Client Sample Sizes** | 49 / 50 / 50 | Exactly **54 / 55 / 54** (Identical to M5.1) | Ad-hoc rounding |
| **Modality Degradation Method**| Metadata sorting/filtering | **Synthetic Feature-Space Transformation** | Feature corruption details |
| **Client 1 Treatment** | Remaining participants | **Clean Audio + Clean Visual + Clean Text** | — |
| **Client 2 Treatment** | Bottom `session_duration_sec` | **Gaussian Noise on Audio Features ($\sigma=0.25$)** | Noise spectrum / SNR |
| **Client 3 Treatment** | Bottom `valid_frame_ratio` | **Visual Temporal Frame Dropout (20% Mean-Imputation)** | Neutral frame definition |
| **Text Modality** | Clean | **Strictly Clean / Unchanged Across All Clients** | — |
| **Validation Cohort** | Local split (10/10/10) | **Standardized Shared Validation Cohort (56)** | Local val leakage |
| **Global Test Cohort** | Fixed 56 hold-out | **Identical Fixed 56 Hold-Out (Clean)** | — |
| **Initial Checkpoint** | Pretrained `best_model.pt` | **M5.1 Round 000 Checkpoint** (`a1c05247...`) | Pretrained weights |
| **FL Rounds / Schedule** | 20 rounds | **50 rounds** (Identical to M5.1) | Variable stopping |
| **Federated Aggregation** | FedAvg | **FedAvg (`weighted_by_key="num-examples"`)** | Default aggregation |
| **Hyperparameters** | AdamW, lr=5e-5 | **AdamW, lr=1e-4, weight_decay=0.01** (M5.1) | Dynamic tuning |

---

## 5. Architectural Decision for M5.4

1. **Protocol Identifier:** Pre-register protocol version `controlled_modality_skew_v1`.
2. **Strict Invariant Maintenance:** Use the canonical M5.1 IID client partitions (`storage/federated/partitions/IID_3Clients` / `DEFAULT_PARTITION_DIR/IID_3Clients`) ensuring exact sample counts (54, 55, 54) and identical participant-to-client mappings.
3. **Dedicated Transformation Module:** Implement `federated/modality_degradation.py` containing deterministic, reproducible per-sample corruption logic using SHA-256 derived seeds and isolated RNGs.
4. **Training-Only Application:** Degradation is strictly applied to local client training loaders; shared validation and global test loaders always receive original, uncorrupted features.
