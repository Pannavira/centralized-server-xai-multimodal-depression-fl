# FED-MIND Phase 4 — Milestone 5.3 Final Scientific Report
## Controlled Non-IID Quantity Skew Experiment

**Scientific Scenario:** Controlled Non-IID Quantity Skew  
**Status:** M5.3 EXECUTION PASSED | M5.3 CONTROL VALIDITY PASSED  
**Run ID:** `FL-QUANTITY-SKEW-20261019-110746-3A4BFF`  
**Baseline Control Run:** `FL-RUN-20261016-130403-45D587` (M5.1 FL-IID Control)  
**Execution Date:** 2026-10-19  
**Framework:** Flower 1.40.0 Message API + PyTorch 2.11.0+cu128 (NVIDIA GeForce RTX 3050 Laptop GPU)  

---

## A. EXECUTIVE SUMMARY

Milestone 5.3 investigates the isolated effect of **heterogeneous client dataset quantities** (Quantity Skew) in Federated Learning for clinical depression detection. Using the canonical FED-MIND research protocol, the 163 training participants were partitioned into a 60% : 30% : 10% distribution across three FL clients:
- **Client 1:** 98 participants (60.12%)
- **Client 2:** 49 participants (30.06%)
- **Client 3:** 16 participants (9.82%)

All other variables—initial global model weights, architecture (`MultimodalBaselineModel`, 7.83M parameters), loss function (`ClassWeightedMSELoss`), optimizer (`AdamW`, $lr=10^{-4}$), scheduler (`CosineAnnealingLR`), training rounds (50), global hold-out test cohort (56 participants), and shared validation cohort (56 participants)—were held strictly identical to the M5.1 IID control baseline.

**Core Scientific Findings:**
1. **Convergence & Error Trade-off:** Quantity Skew led to a modest increase in continuous regression error on the global hold-out test set ($\text{MAE} = 5.6170$ vs $4.9997$ in IID, $\Delta = +0.6173$; $\text{RMSE} = 6.8015$ vs $6.1402$, $\Delta = +0.6613$).
2. **Elevated Clinical Sensitivity:** Screening sensitivity (Recall for $\text{PHQ-8} \ge 10$) increased substantially from **0.6667** to **0.8095** ($\Delta = +0.1428$, $+21.42\%$).
3. **Substantial Reduction in Missed Depression (False Negatives):** The False Negative Rate (FNR) dropped from **33.33%** (7 missed patients) to **19.05%** (only 4 missed patients), catching 3 additional clinically depressed individuals.
4. **Specificity Trade-off:** False positive rate increased ($\text{FPR} = 54.29\%$ vs $25.71\%$), resulting in lower specificity ($0.4571$ vs $0.7429$) and a reduced Macro F1 ($0.5892$ vs $0.7005$).
5. **Client Dominance Mechanism:** Client 1 accounted for a **75.6% share of summed weighted update magnitudes** ($\frac{n_1}{N} \|\Delta w_1\|_2 = 15.60$ vs $4.62$ for C2 and $0.41$ for C3, representing $\frac{15.60}{15.60 + 4.62 + 0.41} = 75.6\%$). This dominance stems from a compounding effect: the $60.12\%$ FedAvg aggregation weight combined with $7\times$ more local gradient descent steps per round (14 steps for C1 vs 2 steps for C3). Note that this metric measures relative scalar update magnitude rather than automatically constituting 75.6% of the net global parameter update vector, because high-dimensional parameter updates across clients may exhibit constructive or destructive vector interference.
6. **Communication Invariant:** Parameter payload transfer was exactly identical to M5.1 ($8,963.01 \text{ MiB} \approx 8.7529 \text{ GiB}$, $\Delta = 0 \text{ bytes}$), confirming that FedAvg bandwidth overhead is independent of local dataset volume.

---

## B. M5.1 CONTROL BASELINE (FL-IID)

The reference benchmark for all comparisons is the official 50-round M5.1 IID experiment (`FL-RUN-20261016-130403-45D587`):

| Parameter | M5.1 Control Value |
| :--- | :--- |
| **Clients / Partition Scheme** | 3 clients / Balanced IID (54 / 55 / 54) |
| **Total Training Cohort** | 163 participants |
| **Server Communication Rounds** | 50 rounds (100% participation) |
| **Local Epochs / Batch Size** | 2 epochs / 16 |
| **Initial Checkpoint SHA-256** | `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973` |
| **Round 50 MAE / RMSE** | 4.9997 / 6.1402 |
| **Round 50 Macro F1 / Recall** | 0.7005 / 0.6667 (TP=14, FN=7) |
| **Round 50 Specificity / ROC-AUC** | 0.7429 / 0.7592 (TN=26, FP=9) |

---

## C. HISTORICAL QUANTITY SKEW AUDIT

An audit of `C:\Wira\Kuliah\Penelitian\clinical-xai-multimodal-depression-fl\federated_learning_output\NonIID_QuantitySkew` demonstrated that the historical partitions could not be reused:
- **Cohort Composition & Overlap Mismatch:** The historical training partition contained 148 unique participants (Client 1: 79, Client 2: 46, Client 3: 23), whereas the M5.1 baseline training pool contains 163 participants (a net difference of 15 in total sample size). However, set-theoretic intersection reveals that only 92 participants overlapped with the M5.1 cohort ($|S_{\text{HIST}} \cap S_{\text{M5.1}}| = 92$). Consequently, **71 out of 163 M5.1 participants were omitted** from the historical cohort ($163 - 92 = 71$), while **56 extraneous participants outside the M5.1 pool were introduced** ($148 - 92 = 56$). The net difference of 15 is merely an aggregate size disparity, not the count of missing subjects.
- **Validation Fragmentation:** Historical code split validation locally (14/9/5) instead of evaluating on the standardized shared validation cohort (56 participants).
- **Ratio Divergence:** Historical partition followed an approximate 10:6:3 ratio (~53.4% : 31.1% : 15.5%) rather than the canonical 60% : 30% : 10% protocol.

Hence, a dedicated deterministic generator (`federated/partitions/quantity_skew.py`) was constructed to enforce experimental controls.

---

## D. QUANTITY PARTITION METHODOLOGY

The 163 M5.1 training participants were allocated deterministically (seed 42) into capacities $[98, 49, 16]$. To substantially minimize confounding from Label Skew:
1. Participants were grouped into ground-truth binary strata: 50 Positives ($\text{PHQ-8} \ge 10$) and 113 Negatives ($\text{PHQ-8} < 10$).
2. Proportional integer assignment was solved under capacity constraints:
   - Client 1: $98 \times (50/163) \approx 30.06 \to \mathbf{30\text{ pos}}, \mathbf{68\text{ neg}}$
   - Client 2: $49 \times (50/163) \approx 15.03 \to \mathbf{15\text{ pos}}, \mathbf{34\text{ neg}}$
   - Client 3: $16 \times (50/163) \approx 4.91 \to \mathbf{5\text{ pos}}, \mathbf{11\text{ neg}}$
   - Check: $30 + 15 + 5 = 50$; $68 + 34 + 11 = 113$; Totals: $98 + 49 + 16 = 163$.
3. Within each stratum, deterministic permutation under seed 42 preserved reproducible ordering.

---

## E. QUANTITY DISTRIBUTION TABLE

| Client | IID N | Quantity Skew N | FedAvg Weight | Positive | Negative | Positive % | Prev. Deviation | Mean PHQ-8 | Std PHQ-8 | Median |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Client 1** | 54 | **98** | 98/163 (60.12%) | 30 | 68 | 30.61% | -0.06% | 6.49 | 6.16 | 5.0 |
| **Client 2** | 55 | **49** | 49/163 (30.06%) | 15 | 34 | 30.61% | -0.06% | 7.04 | 6.02 | 5.0 |
| **Client 3** | 54 | **16** | 16/163 (9.82%) | 5 | 11 | 31.25% | +0.58% | 6.50 | 6.43 | 4.0 |
| **Global Pool**| 163 | **163** | 163/163 (100%) | 50 | 113 | 30.67% | 0.00% | 6.66 | 6.11 | 5.0 |

- **Max/Min Quantity Ratio:** $98 / 16 = \mathbf{6.125\times}$
- **Global Positive Prevalence:** $50 / 163 = 30.6748\%$
- **Mean Pairwise Jensen–Shannon Distance:** $\mathbf{0.003906}$ (calculated via `scipy.spatial.distance.jensenshannon`, returning $\sqrt{\text{JSD}}$; minimal disparity)

---

## F. LABEL-BALANCE ANALYSIS

Prevalence deviation across all three clients is strictly $< 0.6\%$:
- Client 1: $30.6122\%$ ($\Delta = -0.0626\%$)
- Client 2: $30.6122\%$ ($\Delta = -0.0626\%$)
- Client 3: $31.2500\%$ ($\Delta = +0.5752\%$)

Binary Shannon entropies:
- Global: $0.8887 \text{ bits}$
- Client 1: $0.8880 \text{ bits}$
- Client 2: $0.8880 \text{ bits}$
- Client 3: $0.8951 \text{ bits}$

Because binary prevalence discrepancies are minimal (deviations $< 0.6\%$), **Label Skew confounding was substantially minimized and quantitatively assessed**. However, subtle variations in the continuous PHQ-8 severity score distribution (e.g. median and tail shapes) or multimodal acoustic/visual feature characteristics across partitions may still exist, meaning residual feature-level or continuous-distribution skew cannot be entirely ruled out without exhaustive representation-level tests.

---

## G. PHQ-8 CONTINUOUS DISTRIBUTION DIAGNOSTICS

Clinical severity bin distributions (None-minimal 0-4, Mild 5-9, Moderate 10-14, Moderately severe 15-19, Severe 20-24):
- **Global Pool:** 47.2% / 22.1% / 16.0% / 10.4% / 4.3%
- **Client 1:** 48.0% / 21.4% / 16.3% / 10.2% / 4.1%
- **Client 2:** 44.9% / 24.5% / 14.3% / 12.2% / 4.1%
- **Client 3:** 50.0% / 18.8% / 18.8% / 6.2% / 6.2%

Continuous PHQ-8 means across all clients remain within 6.49–7.04 (Global: 6.66), confirming balanced depression severity across all clients.

---

## H. PARTICIPANT POOL AND LEAKAGE AUDIT

- **Total Unique Training Participants:** 163
- **Participant Pool Equality:** $S_{IID} \equiv S_{QUANTITY}$ (163/163 match)
- **Client Disjointness:** Zero participant overlap ($C_1 \cap C_2 = \emptyset$, $C_1 \cap C_3 = \emptyset$, $C_2 \cap C_3 = \emptyset$)
- **Global Hold-Out Test Leakage:** 0 participants (SHA-256: `27ea4f24fb8aa32fcbc79d06ef7d833c74de1abfc5a7fbd1f76a3d7b07884984`)
- **Shared Validation Cohort Leakage:** 0 participants (SHA-256: `54a1dd5c7ea1a5449df2f4e63a623858765662e5d588699b9f4efa3a6a6728ad`)

---

## I. ROUND-0 CHECKPOINT IDENTITY

- **Source Checkpoint:** `storage/federated/runs/FL-RUN-20261016-130403-45D587/rounds/round_000/model.pt`
- **M5.3 Round 0 Checkpoint:** `storage/federated/runs/FL-QUANTITY-SKEW-20261019-110746-3A4BFF/rounds/round_000/model.pt`
- **SHA-256 Checksum:** `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973`
- **Verification Verdict:** **IDENTICAL BITWISE MATCH**

---

## J. GATE A RESULT (PARTITION VALIDATION)

- All 13 pre-condition assertions satisfied.
- Deterministic regeneration test: 100% hash parity.
- **GATE A: PASSED**

---

## K. GATE B RESULT (THREE-ROUND SMOKE TEST)

- Executed 3 full Flower rounds with partitions `[98, 49, 16]`.
- All 3 clients contributed valid model updates.
- Checkpoints `round_000` through `round_003` created and verified.
- Losses remained finite ($>0.0$, no NaN/Inf).
- **GATE B: PASSED**

---

## L. FULL 50-ROUND FLOWER EXECUTION

- **Total Server Rounds Completed:** 50 / 50
- **Total Client Updates Aggregated:** 150 / 150 (3 clients $\times$ 50 rounds, 100% participation)
- **Failures / Dropouts:** 0
- **Total Checkpoints Persisted:** 51 checkpoints (`round_000` to `round_050`)
- **Final Round 50 Checkpoint SHA-256:** `0019ce212413e8956edc71f96d6052fe6a27e374e7963b43c30f20e5b6006e68`

---

## M. CLIENT TRAINING DYNAMICS

Because local batch size is fixed at 16 with 2 local epochs, optimizer step counts naturally differed:

| Client | Training Samples | Batches / Epoch | Optimizer Steps / Round | Mean Train Loss | Mean Train MAE | Mean Local Update Norm $\|\Delta w_i\|$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Client 1** | 98 | 7 | **14 steps** | 23.7261 | 2.6960 | 25.9498 |
| **Client 2** | 49 | 4 | **8 steps** | 28.4061 | 2.8715 | 15.3846 |
| **Client 3** | 16 | 1 | **2 steps** | 26.5899 | 3.6842 | 4.1914 |

---

## N. CLIENT SAMPLE-COUNT WEIGHTING

FedAvg aggregation follows:
$$w_{t+1} = \frac{98}{163} w_1^{(t)} + \frac{49}{163} w_2^{(t)} + \frac{16}{163} w_3^{(t)}$$
- Client 1 weight: **60.12%**
- Client 2 weight: **30.06%**
- Client 3 weight: **9.82%**

Actual client replies returned exact `num-examples` (`[98, 49, 16]`).

---

## O. CLIENT UPDATE / DRIFT ANALYSIS

Weighted update magnitude per client is computed as:
$$\text{Weighted Magnitude}_i = \frac{n_i}{N} \cdot \|\Delta w_i\|_2$$
- **Client 1:** $0.6012 \times 25.9498 = \mathbf{15.6017}$ (**75.6%** share of summed weighted update magnitudes)
- **Client 2:** $0.3006 \times 15.3846 = \mathbf{4.6248}$ (**22.4%** share of summed weighted update magnitudes)
- **Client 3:** $0.0982 \times 4.1914 = \mathbf{0.4114}$ (**2.0%** share of summed weighted update magnitudes)

**Scientific & Mathematical Interpretation:**  
1. **Share of Summed Weighted Update Magnitudes:**  
   The 75.6% figure represents the relative proportion of summed weighted update $L_2$ norms:
   $$\text{Share}_i = \frac{\frac{n_i}{N} \|\Delta w_i\|_2}{\sum_{j=1}^3 \frac{n_j}{N} \|\Delta w_j\|_2} = \frac{15.6017}{15.6017 + 4.6248 + 0.4114} = 75.60\%$$
   Because Client 1 executes 14 local optimizer steps per round (versus 8 for Client 2 and 2 for Client 3), its parameter displacement vector traverses markedly further down the local loss surface ($\|\Delta w_1\|_2 = 25.95$ vs $4.19$). When multiplied by its $60.12\%$ sample aggregation weight, Client 1 produces three-quarters of the summed weighted update magnitude.

2. **Distinction from Net Global Vector Update:**  
   This scalar magnitude share **does not automatically equal 75.6% of the net global update vector**, $\Delta w_{\text{global}} = \sum_{i=1}^3 \frac{n_i}{N} \Delta w_i$. In a 7.83M-dimensional parameter space, update vectors $\Delta w_i$ are spatial directions that can exhibit constructive reinforcement (collinear alignment) or destructive cancellation (opposing gradient components). Thus, the aggregate norm $\|\Delta w_{\text{global}}\|_2$ is generally strictly less than $\sum_{i=1}^3 \frac{n_i}{N} \|\Delta w_i\|_2$ due to the triangle inequality.

3. **Quantifying Directional Contribution to the Global Consensus:**  
   To accurately determine each client's contribution to the *directional trajectory* of the global model, vector projection metrics should be evaluated:
   - **Dot Product / Projection:** The scalar projection $\frac{\langle \frac{n_i}{N} \Delta w_i, \Delta w_{\text{global}} \rangle}{\|\Delta w_{\text{global}}\|_2^2}$, quantifying the signed fraction of global progress directly attributable to client $i$'s vector component.
   - **Cosine Similarity:** $\cos(\theta_i) = \frac{\Delta w_i \cdot \Delta w_{\text{global}}}{\|\Delta w_i\|_2 \|\Delta w_{\text{global}}\|_2}$, quantifying the angular alignment between client $i$'s local update trajectory and the synthesized global consensus update.

---

## P. GLOBAL CONVERGENCE

- Loss decreased rapidly from **196.7155** (Round 0) to **50.62** (Round 10) and converged stably around **40.48** (Round 50).
- Unlike Label Skew where loss oscillated wildly due to conflicting client gradients, Quantity Skew exhibited smooth convergence because all three clients shared approximately identical label distributions (~30.6% depression rate).

---

## Q. IID VS. QUANTITY SKEW PERFORMANCE COMPARISON

Direct comparison between M5.1 Round 50 and M5.3 Round 50 evaluated on the identical 56-participant Global Hold-Out Test set:

| Evaluation Metric | M5.1 IID Control | M5.3 Quantity Skew | Absolute Delta | Relative Delta | Clinical Direction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MAE** | 4.9997 | 5.6170 | +0.6173 | +12.35% | Modest regression error increase |
| **RMSE** | 6.1402 | 6.8015 | +0.6613 | +10.77% | Modest regression error increase |
| **MSE** | 37.7017 | 46.2599 | +8.5582 | +22.70% | Higher variance on extreme scores |
| **Macro F1** | 0.7005 | 0.5892 | -0.1113 | -15.89% | Reduced due to lower specificity |
| **Recall (Sensitivity)** | **0.6667** | **0.8095** | **+0.1428** | **+21.42%** | **Substantially higher depression detection** |
| **Specificity** | 0.7429 | 0.4571 | -0.2858 | -38.47% | More healthy participants flagged |
| **ROC-AUC** | 0.7592 | 0.7293 | -0.0299 | -3.94% | Slight separation reduction |
| **Lin's CCC** | 0.4345 | 0.3268 | -0.1077 | -24.79% | Lower concordance |
| **Pearson r** | 0.4524 | 0.3949 | -0.0575 | -12.71% | Moderate correlation maintained |
| **False Negative Rate (FNR)** | **0.3333** | **0.1905** | **-0.1429** | **-42.87%** | **42.9% relative reduction in missed cases** |
| **False Positive Rate (FPR)** | 0.2571 | 0.5429 | +0.2857 | +111.12% | Increased false alarms |

### Late-Round Stability (Rounds 46–50 Mean $\pm$ Std):
- **MAE:** IID $5.2407 \pm 0.4543$ vs Quantity Skew $5.6223 \pm 0.4103$
- **RMSE:** IID $6.4259 \pm 0.4694$ vs Quantity Skew $6.8033 \pm 0.4581$
- **Macro F1:** IID $0.6635 \pm 0.0641$ vs Quantity Skew $0.5780 \pm 0.0353$
- **Recall:** IID $0.7048 \pm 0.1137$ vs Quantity Skew $0.7714 \pm 0.0398$

Quantity Skew demonstrated lower late-round metric volatility ($\sigma = 0.0353$ for F1 vs $0.0641$ in IID; $\sigma = 0.0398$ for Recall vs $0.1137$ in IID).

---

## R. FALSE NEGATIVE ANALYSIS (CLINICAL SCREENING IMPACT)

In clinical psychiatric screening, a False Negative (failing to identify an individual with clinical depression) carries far greater risk than a False Positive:

| Clinical Metric | M5.1 FL-IID Control | M5.3 Quantity Skew | Delta | Clinical Consequence |
| :--- | :--- | :--- | :--- | :--- |
| **True Positives (TP)** | 14 / 21 | **17 / 21** | **+3** | 3 more depressed patients identified |
| **False Negatives (FN)** | 7 / 21 | **4 / 21** | **-3** | Missed cases reduced by 42.9% |
| **False Negative Rate (FNR)** | 33.33% | **19.05%** | **-14.29%** | Substantial safety enhancement |
| **True Negatives (TN)** | 26 / 35 | 16 / 35 | -10 | Healthy patients correctly confirmed |
| **False Positives (FP)** | 9 / 35 | 19 / 35 | +10 | Healthy patients flagged for review |

Under Quantity Skew, the dominance of Client 1 (which underwent 14 gradient steps per round on a continuous loss with a 2.5x positive weight and 2.0x False Negative penalty) caused the global model to shift toward higher depression sensitivity.

---

## S. COMMUNICATION PAYLOAD COMPARISON

- **Parameters per Model:** 7,832,001 float32 values = 31,328,004 bytes (29.8767 MiB)
- **Round Parameter Payload:** $6 \times 29.8767 \text{ MiB} = 179.2603 \text{ MiB}$
- **Cumulative Logical Payload (50 rounds):**
  - M5.1 Control: **9,398,401,200 bytes** ($8,963.0138\text{ MiB} \approx 8.7529\text{ GiB}$)
  - M5.3 Quantity Skew: **9,398,401,200 bytes** ($8,963.0138\text{ MiB} \approx 8.7529\text{ GiB}$)
  - **Delta:** **0 bytes (0.00%)**

This empirically proves that dataset size heterogeneity among participants does not affect communication bandwidth when client participation remains constant.

---

## T. CHECKPOINT INTEGRITY

- Initial Round 0 Checkpoint SHA-256: `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973` (Matched)
- Final Round 50 Checkpoint SHA-256: `0019ce212413e8956edc71f96d6052fe6a27e374e7963b43c30f20e5b6006e68`
- Strict-load verification with `MultimodalBaselineModel`: **PASSED**
- Production model `storage/models/fl_iid/model.pt` SHA-256: `17d59aa90455386a1e2f00bed2107ef0cbe2a6fb3e96e3f476faf0272db0ec73` (Preserved)

---

## U. PRIVACY VERIFICATION

During all 50 rounds:
- Only `ArrayRecord` weights and safe aggregate scalars (`num-examples`, losses, MAE, update L2 norms) were transmitted.
- Zero raw audio, video, transcripts, patient IDs, or individual gradient records were sent.

---

## V. AUTOMATED TEST SUMMARY

- Comprehensive test suite: **63 passed, 0 failed** in 72.24s.
- Includes 23 dedicated Milestone 5.3 unit and integration tests (`tests/test_federated_quantity_skew.py`).

---

## W. LIMITATIONS AND THREATS TO VALIDITY

1. **Fixed 60:30:10 Ratio:** Evaluated under a single 3-client quantity distribution; more extreme ratios (e.g. 90:5:5) or larger numbers of clients (e.g. 10–50) may exhibit different dynamics.
2. **Deterministic Seed:** Conducted with seed 42 to provide exact reproducibility. Standard deviations reflect intra-run late-round stability, not multi-seed variation.
3. **Natural Optimizer Step Disparity:** Because batch size (16) and local epochs (2) were kept constant, Client 1 executed 14 steps while Client 3 executed 2. This step disparity is an inherent real-world consequence of Quantity Skew.
4. **Clinical Trade-off:** While depression sensitivity improved ($80.95\%$ vs $66.67\%$), specificity dropped ($45.71\%$ vs $74.29\%$), requiring clinical triage workflows to manage false alarms.

---

## X. FINAL STATUS

```
============================================================
FED-MIND — M5.3 NON-IID QUANTITY SKEW
============================================================

Scenario:
Controlled Non-IID Quantity Skew

Baseline:
FL-IID M5.1
FL-RUN-20261016-130403-45D587

Quantity Distribution
---------------------
Client 1: 98 / 163 (60.12%)
Client 2: 49 / 163 (30.06%)
Client 3: 16 / 163 ( 9.82%)

Dataset Validation
------------------
Total training pool : 163
Pool equality       : PASS (S_IID == S_QUANTITY)
Duplicate subjects  : 0
Train/test leakage  : 0
Train/val leakage   : 0
Round-0 hash match  : PASS (a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973)
Label balance       : PASS (Prevalence: C1=30.61%, C2=30.61%, C3=31.25%, Global=30.67%)
Mean JS Distance    : 0.003906

FedAvg Weights
--------------
Client 1: 98/163 (60.12%)
Client 2: 49/163 (30.06%)
Client 3: 16/163 ( 9.82%)

Full Experiment
---------------
Rounds        : 50 / 50
Clients       : 3 / 3
Failures      : 0

Global Evaluation
-----------------
Metric         IID       Quantity Skew      Delta
MAE            4.9997    5.6170             +0.6173 (+12.35%)
RMSE           6.1402    6.8015             +0.6613 (+10.77%)
MSE            37.7017   46.2599            +8.5582 (+22.70%)
Macro F1       0.7005    0.5892             -0.1113 (-15.89%)
Recall         0.6667    0.8095             +0.1428 (+21.42%)
Specificity    0.7429    0.4571             -0.2858 (-38.47%)
ROC-AUC        0.7592    0.7293             -0.0299 (-3.94%)
CCC            0.4345    0.3268             -0.1077 (-24.79%)
Pearson r      0.4524    0.3949             -0.0575 (-12.71%)
FNR            0.3333    0.1905             -0.1429 (-42.87%)
FPR            0.2571    0.5429             +0.2857 (+111.12%)

Client Analysis
---------------
Largest client contribution : 75.6% share of summed weighted update magnitudes (15.60 vs 4.62 vs 0.41)
Client update norms         : C1 = 25.95, C2 = 15.38, C3 = 4.19
Optimizer steps per client  : C1 = 14 steps, C2 = 8 steps, C3 = 2 steps

Communication
-------------
IID payload      : 9,398,401,200 bytes (8,963.01 MiB / 8.7529 GiB)
Quantity payload : 9,398,401,200 bytes (8,963.01 MiB / 8.7529 GiB)
Delta            : 0 bytes (0.00%)

Checkpoint
----------
Round 000 SHA256 : a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973
Round 050 SHA256 : 0019ce212413e8956edc71f96d6052fe6a27e374e7963b43c30f20e5b6006e68

Testing
-------
63 passed
0 failed

============================================================
M5.3 EXECUTION PASSED
M5.3 CONTROL VALIDITY PASSED
============================================================
```
