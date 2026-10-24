# FED-MIND — Milestone 5.5 Dependency & Scientific Consistency Audit
**M5.4 Scientific Cleanup, Metric Re-conciliation, and Resume Reproducibility Audit**  
**Date:** 2026-10-19  
**Status:** COMPLETED & SCIENTIFICALLY RECONCILED

---

## 1. Executive Summary

In preparation for **Milestone 5.5 (Controlled Combined Non-IID)**, a focused audit was conducted on Milestone 5.4 (Modality Skew) analytical outputs, evaluation pipelines, client drift arithmetic, and checkpoint-and-resume mechanisms.

This audit establishes three critical findings:
1. **Primary Clean vs. Supplementary Robustness Clean Evaluation:** Verified bitwise and numerically identical down to $0.0000$ discrepancy ($\Delta = 0.0$).
2. **Client Drift Aggregation Arithmetic:** The reported global client drift ($15.4478$) in M5.4 is mathematically confirmed as the **equal-client arithmetic mean** (and grand pooled mean of 150 client updates). In M5.4, the sample-weighted drift is $15.4494$ ($\Delta = 0.0016$). For M5.5 (where quantities are skewed to 98/49/16), both equal-client mean, sample-weighted mean, and net aggregated update norm will be reported explicitly to prevent conflation.
3. **Resume Reproducibility:** The checkpoint-and-resume implementation preserves global model weights, server round progression, and metric continuity with strict parameter parity ($\max |\mathbf{w}_A - \mathbf{w}_B| = 0.0$).

---

## 2. Clean Evaluation Consistency Audit

### 2.1 Comparative Analysis
M5.4 utilized two independent evaluation execution pathways:
1. **Primary Centralized Evaluation (`GlobalEvaluator`):** Executed at server round completion inside `central_eval_fn()`, operating directly on the standardized 56-subject hold-out test set (`build_global_test_dataloader()`).
2. **Supplementary Robustness Script (`scripts/evaluate_robustness.py`):** Executed post-experiment to assess inference-time degradation under four fixed test conditions (Condition A: Clean, Condition B: Audio Noisy, Condition C: Visual Degraded, Condition D: Combined).

A dedicated script was executed to evaluate the final Round-50 checkpoint of `FL-MODALITY-SKEW-20261019-142242-CA6D11` across both pipelines:

| Metric | Primary Global Evaluation (`GlobalEvaluator`) | Robustness Condition A (`evaluate_robustness.py`) | Numerical Delta ($\Delta$) | Tolerated Threshold | Audit Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MAE** | 4.9739 | 4.9739 | **0.0000** | $< 10^{-4}$ | **EXACT MATCH** |
| **RMSE** | 6.0214 | 6.0214 | **0.0000** | $< 10^{-4}$ | **EXACT MATCH** |
| **MSE** | 36.2572 | 36.2572 | **0.0000** | $< 10^{-4}$ | **EXACT MATCH** |
| **Macro F1** | 0.7214 | 0.7214 | **0.0000** | $< 10^{-4}$ | **EXACT MATCH** |
| **Recall** | 0.7143 | 0.7143 | **0.0000** | $< 10^{-4}$ | **EXACT MATCH** |
| **Specificity** | 0.7429 | 0.7429 | **0.0000** | $< 10^{-4}$ | **EXACT MATCH** |
| **ROC-AUC** | 0.7741 | 0.7741 | **0.0000** | $< 10^{-4}$ | **EXACT MATCH** |
| **Test Loss** | 36.1874 | 36.1874 | **0.0000** | $< 10^{-4}$ | **EXACT MATCH** |
| **TP / FP / TN / FN** | 15 / 9 / 26 / 6 | 15 / 9 / 26 / 6 | **0 / 0 / 0 / 0** | Exact integer | **EXACT MATCH** |

### 2.2 Root Cause of Prior Perceived Discrepancy
In the M5.4 draft report, an informal late-round stability mean ($\text{MAE} = 5.4441$, $\text{RMSE} = 6.5135$, averaged across rounds 46–50) was juxtaposed near the Round-50 final checkpoint row, creating superficial confusion in preliminary reading. The final Round-50 checkpoint metrics and the supplementary robustness Condition A metrics are strictly identical across all statistical and clinical dimensions.

---

## 3. Client Drift Consistency Audit

### 3.1 Mathematical Decomposition
Client drift in FED-MIND measures the Euclidean parameter divergence between the incoming global model $w_{r}$ and the updated local model $w_{k, r}$ after local optimization:
$$\|\Delta w_{k, r}\|_2 = \|w_{k, r} - w_{r}\|_2$$

In M5.4, the reported per-client update $L_2$ norms across all 50 rounds were:
- **Client 1 (Clean, $N_1=54$):** Mean $\|\Delta w_1\|_2 = 15.2868$
- **Client 2 (Audio Noisy, $N_2=55$):** Mean $\|\Delta w_2\|_2 = 15.7089$
- **Client 3 (Visual Degraded, $N_3=54$):** Mean $\|\Delta w_3\|_2 = 15.3476$

### 3.2 Audit of Global Reported Drift
The reported global value of $15.4478$ is derived as:
$$\text{Equal-Client Mean} = \frac{15.2868 + 15.7089 + 15.3476}{3} = \frac{46.3433}{3} = 15.447766\dots \approx \mathbf{15.4478}$$
This also strictly equals the grand pooled mean across all 150 client updates:
$$\text{Grand Pooled Mean} = \frac{1}{50 \times 3} \sum_{r=1}^{50} \sum_{k=1}^3 \|\Delta w_{k, r}\|_2 = \frac{2317.1649}{150} = \mathbf{15.4478}$$

The **sample-weighted drift** for M5.4 is:
$$\text{Sample-Weighted Mean} = \sum_{k=1}^3 \frac{N_k}{N} \overline{\|\Delta w_k\|_2} = \frac{54(15.2868) + 55(15.7089) + 54(15.3476)}{163} = \frac{2518.2471}{163} = \mathbf{15.4494}$$
Because $N_1=54, N_2=55, N_3=54$, the sample weights ($\approx 0.3313, 0.3374, 0.3313$) are virtually equal, resulting in a negligible delta of $0.0016$ ($0.01\%$).

### 3.3 Rule for M5.5 Combined Non-IID
In M5.5, sample sizes are intentionally asymmetric ($98 / 49 / 16$ or $60.12\% / 30.06\% / 9.82\%$). The equal-client mean and sample-weighted mean will diverge substantially. Therefore, M5.5 will explicitly report:
1. **Raw Per-Client Mean Drift:** $\overline{\|\Delta w_k\|_2}$ for $k \in \{1, 2, 3\}$.
2. **Equal-Client Mean:** $\frac{1}{3} \sum_{k=1}^3 \overline{\|\Delta w_k\|_2}$.
3. **Sample-Weighted Mean Drift:** $\sum_{k=1}^3 \frac{N_k}{163} \overline{\|\Delta w_k\|_2}$.
4. **Net Aggregated Global Update Norm:** $\|\Delta w_{\text{global}}\|_2 = \left\|\sum_{k=1}^3 \frac{N_k}{163} \Delta w_{k, r}\right\|_2$.

---

## 4. Resume Reproducibility Audit

### 4.1 Invariance Test Findings
The automated test `test_resume_equivalence_invariance` executes an identical local training step across two pathways:
- **Path A (Continuous):** Model initialized from state $w_0$ and optimized for 1 epoch.
- **Path B (Save & Reload):** State $w_0$ serialized to disk, reloaded into a new model instance, and optimized with identical inputs.
- **Result:** $\max |\mathbf{w}_A - \mathbf{w}_B| = 0.0$ strictly across all 7,832,001 parameters.

### 4.2 Multi-Round Federated Resume Characteristics
When a federated run resumes from an intermediate round $R_{\text{resume}}$:
1. **Global Model State:** Loaded strictly from `round_{R:03d}/model.pt`.
2. **Round Counter:** Restored to $R_{\text{resume}} + 1$.
3. **Metric Continuity:** Historical entries in `metrics.csv`, `client_metrics.csv`, and `communication.csv` are preserved; new rounds append directly.
4. **Optimizer & Scheduler:** Under FedAvg, client optimizers (`AdamW`) and learning rate schedulers (`CosineAnnealingLR`) are instantiated fresh per client round, matching the standard FL local-epoch protocol.
5. **Deterministic Corruption:** Feature corruption uses stable SHA-256 seed hashing derived from participant ID and modality (`get_deterministic_sample_seed`), ensuring that feature corruption is strictly invariant across process restarts.
6. **Stochastic Ordering:** Standard PyTorch DataLoader shuffle uses process-level RNG. While parameter updates are smooth and bounded, slight numerical variance in DataLoader batch ordering across process boundaries is expected and documented.

---

## 5. Audit Conclusion

The analytical foundations of Milestone 5.4 are verified as sound, reproducible, and internally consistent. With the distinction between equal-client mean drift and sample-weighted drift explicitly codified, the system is fully prepared for Milestone 5.5 Combined Non-IID execution.
