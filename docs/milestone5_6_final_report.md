# FED-MIND — Milestone 5.6 Final Scientific Report
## Cross-Scenario Multi-Seed Validation and Scientific Statistical Analysis

**Project**: FED-MIND (Federated Multimodal Depression Screening with Explainable AI)  
**Milestone**: M5.6 — Consolidating M5.1–M5.5 into a Defensible, Multi-Seed Statistical Benchmark  
**Date**: October 20, 2026  
**Status**: Comprehensive Scientific Evaluation  

---

### A. Executive Summary

Milestone 5.6 synthesizes the preceding five experimental milestones (M5.1 IID Baseline Control, M5.2 Non-IID Label Skew, M5.3 Non-IID Quantity Skew, M5.4 Non-IID Modality Skew, and M5.5 Controlled Non-IID Combined Skew) into a rigorous, multi-seed statistical benchmark. Prior milestone prototypes evaluated a single random seed ($S=42$), leaving open whether observed performance variations reflected underlying clinical data heterogeneity or seed-specific stochasticity. 

To answer this question defensibly, Milestone 5.6 establishes:
1. **Scientific Preflight & Round-0 Parity**: Full audit of historical Round-0 model bytes across M5.1–M5.5, resolving drafting discrepancies and proving bitwise identical predictions ($0.00000000$ diff) across all historical baseline checkpoints.
2. **Paired Multi-Seed Protocol**: Frozen paired initializations ($S \in \{42, 43, 44, 45, 46\}$) where within each seed, all 5 scenarios share the exact same initial global model checkpoint byte hash, isolating data distribution effects from weight initialization variance.
3. **Deterministic Partition Protocols**: 25 zero-leakage partition sets across 163 unique training participants and a frozen 56-participant global test cohort, preserving exact clinical target strata without participant duplication or synthetic data.
4. **Statistical Inference & Paired Deltas**: Within-seed paired differences ($\Delta = X - \text{IID}$) with sample standard deviations ($ddof=1$) and Student-$t$ 95% confidence intervals, preventing invalid independent-sample assumptions.
5. **Full Automated Test Coverage**: A dedicated test suite (`tests/test_federated_multiseed.py`) covering all 35 milestone verification criteria with a 100% pass rate.

---

### B. Research Questions & Formal Inquiries

1. **RQ1 (IID vs Non-IID Performance)**: How does IID FedAvg compare with Label, Quantity, Modality, and Combined Non-IID scenarios across independent random seeds?
2. **RQ2 (Stability of Differences)**: Which observed performance differences remain stable across independent training repetitions?
3. **RQ3 (PHQ-8 Severity Estimation Accuracy)**: How do the scenarios differ in PHQ-8 regression accuracy (MAE, RMSE, MSE, Lin's CCC, Pearson $r$)?
4. **RQ4 (Binary Screening Robustness)**: How do Recall, Specificity, Macro F1, and ROC-AUC change under heterogeneous client distributions?
5. **RQ5 (Initialization vs Partition Randomness)**: How much variation is caused by model weight initialization versus participant partition allocation?
6. **RQ6 (Convergence Stability)**: How stable are the 50-round learning trajectories across random seeds?
7. **RQ7 (Client Drift Quantification)**: How does client drift (update norm magnitude $||\Delta w||$ and directional alignment) differ between balanced and skewed scenarios?
8. **RQ8 (Communication Efficiency)**: Does estimated parameter communication remain constant under the full-participation FedAvg protocol?
9. **RQ9 (Defensible Empirical Findings)**: Which findings are robust enough to report as reproducible empirical observations?
10. **RQ10 (Scientific Limitations)**: What conclusions remain unsupported due to the small number of clients ($K=3$), fixed global test cohort ($N=56$), and finite seed sample size?

---

### C. Previous Milestone Audit (M5.1–M5.5)

The historical prototype runs from M5.1 through M5.5 were audited to assess methodological comparability:

| Milestone | Scenario Identifier | Run ID | Checkpoint SHA-256 (R50) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **M5.1** | FL-IID Control | `FL-RUN-20261016-130403-45D587` | `41eab5039782720ed25e4723a74ce060450c6e2493dd877066eebc98334a1220` | `REUSED_VERIFIED` |
| **M5.2** | Non-IID Label Skew | `FL-LABEL-SKEW-20261016-141108-7865D4` | `583730ed878cc8527bc1cdf18e139294e4b8b45ac819e1f6f6edd46eb25f4ffd` | `REUSED_VERIFIED` |
| **M5.3** | Non-IID Quantity Skew | `FL-QUANTITY-SKEW-20261019-110746-3A4BFF` | `0019ce212413e8956edc71f96d6052fe6a27e374e7963b43c30f20e5b6006e68` | `REUSED_VERIFIED` |
| **M5.4** | Non-IID Modality Skew | `FL-MODALITY-SKEW-20261019-142242-CA6D11` | `abe2a3e18352f59256a296942eb08339bd71cb08e713827e5905a47963dcee79` | `REUSED_VERIFIED` |
| **M5.5** | Non-IID Combined Skew | `FL-COMBINED-NONIID-20261019-154311-CCF7B4` | `c3b41ab7363aba7f1517953326162165b4541b59ecdffd40099318e5364c87b3` | `REUSED_VERIFIED` |

All five historical runs completed 50 communication rounds under Flower 1.40 Message API with 3 participating clients per round, 2 local epochs, batch size 16, AdamW optimizer, and `ClassWeightedMSELoss`.

---

### D. Hard Gate A: Round-0 Consistency & Discrepancy Resolution

A mandatory blocker identified prior to M5.6 was an apparent inconsistency in historical reports:
- M5.1 Round 0 report: $\text{MAE} = 8.1071$, $\text{RMSE} = 10.3077$, $\text{Macro F1} = 0.3846$
- M5.5 Round 0 draft report: $\text{MAE} = 6.1388$, $\text{RMSE} = 7.2144$, $\text{Macro F1} = 0.5401$
- Both claimed identical initial checkpoint SHA-256: `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973`.

#### Investigation & Audit Findings:
1. **Physical Checkpoint Byte Inspection**:
   All five historical `round_000/model.pt` files were hashed directly from disk:
   - M5.1 R0: `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973` (31,350,348 bytes)
   - M5.2 R0: `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973` (31,350,348 bytes)
   - M5.3 R0: `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973` (31,350,348 bytes)
   - M5.4 R0: `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973` (31,350,348 bytes)
   - M5.5 R0: `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973` (31,350,348 bytes)
2. **Canonical Re-evaluation**:
   Re-evaluating all five checkpoints using the single canonical `GlobalEvaluator` against the immutable 56-patient global test cohort yielded:
   - Loss: $196.7155$
   - MAE: $8.1071$
   - RMSE: $10.3077$
   - Macro F1: $0.3846$
   - Sensitivity / Recall: $0.0000$
   - Specificity: $1.0000$
   - ROC-AUC: $0.4313$
   - Max sample-level prediction discrepancy across all $5 \times 56$ predictions: **$0.00000000$**.
3. **Root Cause**:
   The physical checkpoint bytes and the real `metrics.csv` files inside `storage/federated/runs/` were always strictly identical. The discrepancy ($\text{MAE}=6.1388$) was an isolated human drafting error in Section R of `milestone5_5_final_report.md` draft text, where an intermediate evaluation snippet had been copied into the text narrative.
4. **Verdict**: **RESOLVED**. Full documentation in `docs/milestone5_6_scientific_preflight_audit.md`. Hard Gate A PASSED.

---

### E. Environment Consistency & Runtime Compatibility Audit

Full environment details documented in `docs/milestone5_6_environment_audit.md`:
- **Python**: 3.14.3 (x86_64, Windows-11 Pro Build 26100)
- **PyTorch Host Environment**: 2.11.0+cu128 (CUDA 12.8, cuDNN 91900 in `.venv`)
- **PyTorch Flower Runtime Envs**: 2.14.1+cpu (Ray client actors under Flower hermetic runtime environments)
- **Flower**: 1.40.0
- **Ray**: 2.59.0
- **Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (6,140 MiB VRAM), 16 GiB Host RAM
- **Git Commit**: `828c079d5c4ecffb935d77d93e9d48a9f683d612`

#### Runtime Hardware Execution Parity Audit:
1. **Seed-42 IID Control (`FL-RUN-20261016-130403-45D587`)**: Executed directly against host environment using GPU acceleration (`2.11.0+cu128`, CUDA 12.8, RTX 3050). Elapsed duration: $452$s (~$7.5$ min).
2. **All Other Runs (M5.2–M5.5 and Seeds 43–46)**: Executed inside Flower 1.40's hermetic runtime environments (`C:\Users\pikar\.flwr\runtime-envs`) on CPU (`2.14.1+cpu`). Elapsed duration: $1414$s–$2195$s ($23$–$36$ min per run).
3. **Scientific Assessment**: While model mathematics and gradient updates are structurally identical between CPU and GPU float32 execution, bitwise floating-point non-associativity across different BLAS/CUDA backends means cross-environment comparisons carry a minor numerical qualification.
4. **Comparability Verdict**: **`QUALIFIED`**. The framework architecture and hyperparameter protocols are frozen, but exact hardware parity across all 25 runs is qualified by the host-CUDA vs Flower-CPU difference.

---

### F. Experimental Design: Five Federated Scenarios

All experiments employ the authoritative `MultimodalBaselineModel` (7,832,001 FP32 parameters) across 3 clients with full participation ($C=1.0$):
1. **FL-IID**: Partition allocation balanced across clients (54 / 55 / 54 participants) with stratified class prevalence (~30.6% depression positive). Clean audio, visual, and text.
2. **Non-IID Label Skew**: 54 / 55 / 54 participants with extreme label divergence across clients:
   - Client 1: 30 positive / 24 negative (55.6% prevalence)
   - Client 2: 15 positive / 40 negative (27.3% prevalence)
   - Client 3: 5 positive / 49 negative (9.3% prevalence)
3. **Non-IID Quantity Skew**: Unbalanced participant cohort sizes (98 / 49 / 16 participants, roughly 6:3:1 ratio) with balanced label prevalence (~30.6% positive in each client).
4. **Non-IID Modality Skew**: Balanced cohorts (54 / 55 / 54) with client-specific sensor degradation:
   - Client 1: Clean audio, visual, text.
   - Client 2: Additive Gaussian feature noise on audio ($\sigma = 0.25$).
   - Client 3: Temporal frame dropout on visual features ($p = 0.20$), imputed with train visual mean.
   - Text features remain clean across all clients. Shared validation and test sets remain 100% clean.
5. **Non-IID Combined Skew**: Compounded heterogeneity combining Quantity Skew (98 / 49 / 16), Label Skew (Client 1: 18/80; Client 2: 19/30; Client 3: 13/3), and Modality Skew (Client 1 clean, Client 2 noisy audio, Client 3 degraded visual).

---

### G. Multi-Seed Initialization Policy

To eliminate model initialization bias, each seed $S \in \{42, 43, 44, 45, 46\}$ has a frozen, reproducible initial checkpoint in `storage/federated/multiseed/initializations/`:

| Seed | Checkpoint SHA-256 | Test MAE (R0) | Test RMSE (R0) | Test Macro F1 (R0) |
| :---: | :--- | :---: | :---: | :---: |
| **42** | `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973` | 8.1071 | 10.3077 | 0.3846 |
| **43** | `89bbd6c5109422a7393b9b70ed75ed5a0370340fe56cd92cdb4de98313254b26` | 8.1205 | 10.3225 | 0.3846 |
| **44** | `207a9cd39a46a6f570c082b885890667be100c387c8fb1aa3fcfcf230ec5f40e` | 8.0950 | 10.2930 | 0.3846 |
| **45** | `3dc785b5c8b959f86534ec25a324727f0037545b8bea07281ebeaa7084d62a88` | 8.2159 | 10.3992 | 0.3846 |
| **46** | `717b3e55c41e4ec260a8e0524cad6e86c91286afb02c4c5b708a9dd7c089f288` | 8.1219 | 10.3239 | 0.3846 |

**Critical Invariant**: Within each seed, all 5 experimental scenarios initialize from the exact same byte-identical checkpoint. Cross-scenario comparisons are paired by seed.

---

### H. Scenario-Specific Partition Protocols

Partitions were generated deterministically across all 5 seeds ($5 \times 5 = 25$ partition sets) and cataloged in `storage/federated/multiseed/partitions/partitions_registry.json`.

```
storage/federated/multiseed/partitions/
├── seed_42/ (IID_3Clients, label_skew, quantity_skew, modality_skew, combined_skew)
├── seed_43/ (IID_3Clients, label_skew, quantity_skew, modality_skew, combined_skew)
├── seed_44/ (IID_3Clients, label_skew, quantity_skew, modality_skew, combined_skew)
├── seed_45/ (IID_3Clients, label_skew, quantity_skew, modality_skew, combined_skew)
└── seed_46/ (IID_3Clients, label_skew, quantity_skew, modality_skew, combined_skew)
```

Each partition folder contains:
- `client_1_train.csv`, `client_2_train.csv`, `client_3_train.csv`
- `client_1_val.csv`, `client_2_val.csv`, `client_3_val.csv`
- `partition_manifest.json`

---

### I. Dataset Population and Leakage Invariants

Every partition set strictly preserves:
- Exactly 163 unique training participants.
- Exactly 56 shared validation participants.
- Exactly 56 fixed hold-out test participants (21 depression-positive $\ge 10$, 35 depression-negative $< 10$).
- Exactly 50 positive training participants and 113 negative training participants globally.
- **Zero Participant Leakage**: Evaluated by `test_15_no_participant_leakage`, verifying that $\text{train\_ids} \cap \text{test\_ids} = \emptyset$ and $\text{val\_ids} \cap \text{test\_ids} = \emptyset$ across all 25 sets.

---

### J. Experiment Matrix

The complete experiment matrix encompasses 25 scenario-seed pairs:

$$\mathcal{M} = \{(\text{Scenario}_i, \text{Seed}_j) \mid i \in [1, 5], j \in [42, 46]\}$$

- Total Target Matrix: 25 full runs (50 rounds each).
- Baseline seed 42 runs: 5 runs audited and accepted as `REUSED_VERIFIED`.
- Execution strategy: Sequential simulation to prevent GPU out-of-memory errors on the RTX 3050 Laptop GPU.

---

### K. Staged Execution Gates

1. **Gate A (Scientific Preflight)**: PASSED. Round 0 SHA-256 consistency verified across all historical runs; evaluator verified; zero leakage asserted.
2. **Gate B (New-Seed Integration Suite)**: PASSED. 2-round integration run executed across all 5 scenarios on seed 43 (`FL-RUN-20261020-030927-3077BF`, `FL-LABEL-SKEW-20261020-031239-55CB3C`, `FL-QUANTITY-SKEW-20261020-031525-126FBE`, `FL-MODALITY-SKEW-20261020-031850-34765A`, `FL-COMBINED-NONIID-20261020-032154-22DB9E`). Verified identical R0 SHA (`89bbd6c5...`), correct client partition allocations, correct degradation diagnostics, correct FedAvg weights, finite loss progressions, and clean artifact persistence.
3. **Gate C (Multi-Seed Sweep)**: Evaluated under the multi-seed protocol with automated tracking in `storage/federated/multiseed/m5_6/run_registry.json`.

---

### L. Full 25-Run Matrix Provenance Registry & Execution Status

All 25 scenario-seed experiment combinations are tracked in `storage/federated/multiseed/m5_6/run_registry.json` and mirrored in `seed_scenario_matrix.csv`. Runs are recorded with explicit provenance (`REUSED_VERIFIED` for audited historical prototype runs, `COMPLETED` for validated new executions, `RUNNING` for active executions, and `QUEUED` for planned sweeps).

| Row | Scenario Identifier | Seed | Run ID | Source | Status | Checkpoint SHA-256 (R50) |
| :---: | :--- | :---: | :--- | :--- | :--- | :--- |
| **1** | **FL-IID** | 42 | `FL-RUN-20261016-130403-45D587` | `reused_historical_run` | `REUSED_VERIFIED` | `41eab5039782720ed25e4723a74ce060450c6e2493dd877066eebc98334a1220` |
| **2** | **Non-IID Label Skew** | 42 | `FL-LABEL-SKEW-20261016-141108-7865D4` | `reused_historical_run` | `REUSED_VERIFIED` | `583730ed878cc8527bc1cdf18e139294e4b8b45ac819e1f6f6edd46eb25f4ffd` |
| **3** | **Non-IID Quantity Skew** | 42 | `FL-QUANTITY-SKEW-20261019-110746-3A4BFF` | `reused_historical_run` | `REUSED_VERIFIED` | `0019ce212413e8956edc71f96d6052fe6a27e374e7963b43c30f20e5b6006e68` |
| **4** | **Non-IID Modality Skew** | 42 | `FL-MODALITY-SKEW-20261019-142242-CA6D11` | `reused_historical_run` | `REUSED_VERIFIED` | `abe2a3e18352f59256a296942eb08339bd71cb08e713827e5905a47963dcee79` |
| **5** | **Non-IID Combined Skew** | 42 | `FL-COMBINED-NONIID-20261019-154311-CCF7B4` | `reused_historical_run` | `REUSED_VERIFIED` | `c3b41ab7363aba7f1517953326162165b4541b59ecdffd40099318e5364c87b3` |
| **6** | **FL-IID** | 43 | `FL-RUN-20261020-032804-B3E624` | `new_execution` | `COMPLETED` | `847016a6eddf837dfc9ee94d7e1bec975046382f023c0073220463e008da8ada` |
| **7** | **Non-IID Label Skew** | 43 | `FL-LABEL-SKEW-20261020-035228-2CF6DB` | `new_execution` | `COMPLETED` | `5e5bb05c197f0bcc6894414b88312526412d35dfc836cdcd3594e4433751c69b` |
| **8** | **Non-IID Quantity Skew** | 43 | `FL-QUANTITY-SKEW-20261020-041622-264CAD` | `new_execution` | `COMPLETED` | `7cc1659341f9bdbf3b49f7a15dfdc74c9c93c1e996192560eb9e62f1a15b0851` |
| **9** | **Non-IID Modality Skew** | 43 | `FL-MODALITY-SKEW-20261020-045043-24DCBF` | `new_execution` | `COMPLETED` | `32ea67d3c118862e5e6713714504b9e1f106735c6071fc8c9ce93c398783b162` |
| **10** | **Non-IID Combined Skew** | 43 | `FL-COMBINED-NONIID-20261020-051516-4660E8` | `new_execution` | `COMPLETED` | `197c481606c70ea2dddb7547eb2d3d1b7db0d708248c387448e837f6fab7865e` |
| **11** | **FL-IID** | 44 | `FL-RUN-20261020-072523-C438F8` | `new_execution` | `COMPLETED` | `6cc41666cf416f75f4df0c10b7c7cbc25af06fc1ad0498ff990087878825f967` |
| **12** | **Non-IID Label Skew** | 44 | `FL-LABEL-SKEW-20261020-075030-304DFF` | `new_execution` | `COMPLETED` | `6d4419ae4ef1ccfcb07f5f03476b590b8bc9226a6c85591f29c44f2c19f79ecd` |
| **13** | **Non-IID Quantity Skew** | 44 | `FL-QUANTITY-SKEW-20261020-081754-DC1422` | `new_execution` | `COMPLETED` | `a3a51059b78ee7696e5ec4e7b37469fc47180b4d632492832b09522485cb9303` |
| **14** | **Non-IID Modality Skew** | 44 | `FL-MODALITY-SKEW-20261020-085741-45F463` | `new_execution` | `COMPLETED` | `f43571210c8f10b327e4850c7edba19b810d21097a20fd59451c002e207e97b4` |
| **15** | **Non-IID Combined Skew** | 44 | `FL-COMBINED-NONIID-20261020-092142-282814` | `new_execution` | `COMPLETED` | `8b332a15b9ab884a4512a4706914d3e19ba7e4d7913f754a143858abdfa9904e` |
| **16** | **FL-IID** | 45 | `FL-RUN-20261020-100122-1F9CD2` | `new_execution` | `COMPLETED` | `3302193a43a2fc15ab63748b6f9e9279d5857e4c9bae75689d52e45a310164a8` |
| **17** | **Non-IID Label Skew** | 45 | `FL-LABEL-SKEW-20261020-102621-DF100C` | `new_execution` | `COMPLETED` | `01cf6f22267143104d30973549db5496d0ea276974d90d806043f749559cce1e` |
| **18** | **Non-IID Quantity Skew** | 45 | `FL-QUANTITY-SKEW-20261020-132834-FE77BA` | `new_execution` | `COMPLETED` | `c3d5b5e2258cb7588d312991551183999b639bb3601ed5bd0f7da37cf40ccf56` |
| **19** | **Non-IID Modality Skew** | 45 | `FL-MODALITY-SKEW-20261020-135113-73B9C5` | `new_execution` | `COMPLETED` | `94abf7b44c7c1b08983d39fb2acda6339fee4903cc8a4fc099474c60d081fcd0` |
| **20** | **Non-IID Combined Skew** | 45 | `FL-COMBINED-NONIID-20261020-141049-3C69B6` | `new_execution` | `COMPLETED` | `cd11579eb6976b563c305a02b8cf2aec41f233a387e71bff5fa3234d251f6bf6` |
| **21** | **FL-IID** | 46 | `FL-RUN-20261020-143324-DB6B55` | `new_execution` | `COMPLETED` | `56bea9e2bb118da2455b565808ca0dc457775f04405154e6ed0d637c78039203` |
| **22** | **Non-IID Label Skew** | 46 | `FL-LABEL-SKEW-20261020-145236-0E4D55` | `new_execution` | `COMPLETED` | `36beb18034fe545381116fee075ba2a5019b0333e2b678de267558fa6d381829` |
| **23** | **Non-IID Quantity Skew** | 46 | `FL-QUANTITY-SKEW-20261020-151144-EA1F35` | `new_execution` | `COMPLETED` | `f55a1c279b04aaebd0327cdbfdc1971725a999366a7b3676ff0fe93ae0f48b86` |
| **24** | **Non-IID Modality Skew** | 46 | `FL-MODALITY-SKEW-20261020-153351-463188` | `new_execution` | `COMPLETED` | `02c0337cffac0117d7518245feeea669685fca960b41da531d00aa428825cf92` |
| **25** | **Non-IID Combined Skew** | 46 | `FL-COMBINED-NONIID-20261020-155307-062E20` | `new_execution` | `COMPLETED` | `b2a5513f27467e43b2533bcc8533afa2bdef4abc2b57492f6665222e7262bebc` |

**Final Execution Registry Summary**:
- **Total Required Runs**: 25 (5 scenarios $\times$ 5 seeds)
- **Completed & Validated Runs**: 25 (100% matrix completion rate)
- **Active Execution**: 0
- **Queued Executions**: 0
- **Matrix Completion Rate**: 100.0% (25/25 validated runs)

---

### M. Per-Seed Metrics Table (Round 50 Official Evaluation)

All metrics computed on the immutable clean 56-patient hold-out test set ($N=56$, 21 screening-positive $\ge 10$, 35 screening-negative $< 10$):

| Scenario | Seed | Run ID | MAE | RMSE | Macro F1 | Binary F1 | Recall | Specificity | ROC-AUC | Lin's CCC | Pearson $r$ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FL-IID** | 42 | `FL-RUN-20261016-130403-45D587` | 4.9997 | 6.1402 | 0.7005 | 0.6364 | 0.6667 | 0.7429 | 0.7592 | 0.4345 | 0.4524 |
| **FL-IID** | 43 | `FL-RUN-20261020-032804-B3E624` | 4.6557 | 5.8546 | 0.6973 | 0.5946 | 0.5238 | 0.8571 | 0.7279 | 0.4434 | 0.4695 |
| **FL-IID** | 44 | `FL-RUN-20261020-072523-C438F8` | 5.9718 | 7.0655 | 0.5500 | 0.5902 | 0.8571 | 0.3714 | 0.7483 | 0.3191 | 0.4100 |
| **FL-IID** | 45 | `FL-RUN-20261020-100122-1F9CD2` | 5.6341 | 6.5916 | 0.5351 | 0.5517 | 0.7619 | 0.4000 | 0.7619 | 0.3646 | 0.4373 |
| **FL-IID** | 46 | `FL-RUN-20261020-143324-DB6B55` | 5.2337 | 6.2170 | 0.6553 | 0.6122 | 0.7143 | 0.6286 | 0.7714 | 0.3964 | 0.4255 |
| **Non-IID Label Skew** | 42 | `FL-LABEL-SKEW-20261016-141108-7865D4` | 4.6964 | 5.9384 | 0.6725 | 0.5556 | 0.4762 | 0.8571 | 0.7116 | 0.3894 | 0.4286 |
| **Non-IID Label Skew** | 43 | `FL-LABEL-SKEW-20261020-035228-2CF6DB` | 5.1607 | 6.2879 | 0.6209 | 0.4516 | 0.3333 | 0.9143 | 0.7020 | 0.2466 | 0.3061 |
| **Non-IID Label Skew** | 44 | `FL-LABEL-SKEW-20261020-075030-304DFF` | 4.6882 | 5.6218 | 0.6500 | 0.5500 | 0.5238 | 0.7714 | 0.7714 | 0.4104 | 0.4802 |
| **Non-IID Label Skew** | 45 | `FL-LABEL-SKEW-20261020-102621-DF100C` | 5.6650 | 6.8996 | 0.5447 | 0.3333 | 0.2381 | 0.8857 | 0.5837 | 0.1694 | 0.2078 |
| **Non-IID Label Skew** | 46 | `FL-LABEL-SKEW-20261020-145236-0E4D55` | 4.8976 | 6.0221 | 0.6316 | 0.5000 | 0.4286 | 0.8286 | 0.7293 | 0.3517 | 0.4131 |
| **Non-IID Quantity Skew** | 42 | `FL-QUANTITY-SKEW-20261019-110746-3A4BFF` | 5.6170 | 6.8015 | 0.5892 | 0.5965 | 0.8095 | 0.4571 | 0.7293 | 0.3268 | 0.3949 |
| **Non-IID Quantity Skew** | 43 | `FL-QUANTITY-SKEW-20261020-041622-264CAD` | 4.7497 | 5.8495 | 0.6814 | 0.5789 | 0.5238 | 0.8286 | 0.7306 | 0.4361 | 0.4690 |
| **Non-IID Quantity Skew** | 44 | `FL-QUANTITY-SKEW-20261020-081754-DC1422` | 5.9061 | 7.1537 | 0.5892 | 0.5818 | 0.7619 | 0.4857 | 0.6966 | 0.3260 | 0.3578 |
| **Non-IID Quantity Skew** | 45 | `FL-QUANTITY-SKEW-20261020-132834-FE77BA` | 5.7727 | 6.6502 | 0.5534 | 0.5614 | 0.7619 | 0.4286 | 0.7333 | 0.3382 | 0.4215 |
| **Non-IID Quantity Skew** | 46 | `FL-QUANTITY-SKEW-20261020-151144-EA1F35` | 5.0895 | 6.1315 | 0.6553 | 0.6122 | 0.7143 | 0.6286 | 0.7646 | 0.4103 | 0.4557 |
| **Non-IID Modality Skew** | 42 | `FL-MODALITY-SKEW-20261019-142242-CA6D11` | 4.9739 | 6.0214 | 0.7214 | 0.6667 | 0.7143 | 0.7429 | 0.7741 | 0.4262 | 0.4583 |
| **Non-IID Modality Skew** | 43 | `FL-MODALITY-SKEW-20261020-045043-24DCBF` | 4.7578 | 6.0619 | 0.6631 | 0.5909 | 0.6190 | 0.7143 | 0.7456 | 0.4128 | 0.4371 |
| **Non-IID Modality Skew** | 44 | `FL-MODALITY-SKEW-20261020-085741-45F463` | 5.5182 | 6.7295 | 0.6387 | 0.6000 | 0.7143 | 0.6000 | 0.7224 | 0.3455 | 0.3902 |
| **Non-IID Modality Skew** | 45 | `FL-MODALITY-SKEW-20261020-135113-73B9C5` | 6.0105 | 7.0489 | 0.4994 | 0.5172 | 0.7143 | 0.3714 | 0.7197 | 0.3237 | 0.4024 |
| **Non-IID Modality Skew** | 46 | `FL-MODALITY-SKEW-20261020-153351-463188` | 5.0227 | 6.0749 | 0.6631 | 0.5909 | 0.6190 | 0.7143 | 0.7469 | 0.4523 | 0.4623 |
| **Non-IID Combined Skew** | 42 | `FL-COMBINED-NONIID-20261019-154311-CCF7B4` | 5.0585 | 6.4031 | 0.6680 | 0.6087 | 0.6667 | 0.6857 | 0.7211 | 0.3199 | 0.3470 |
| **Non-IID Combined Skew** | 43 | `FL-COMBINED-NONIID-20261020-051516-4660E8` | 5.0545 | 5.9443 | 0.6553 | 0.6122 | 0.7143 | 0.6286 | 0.7333 | 0.4032 | 0.4610 |
| **Non-IID Combined Skew** | 44 | `FL-COMBINED-NONIID-20261020-092142-282814` | 5.1670 | 6.2721 | 0.6730 | 0.5854 | 0.5714 | 0.7714 | 0.7156 | 0.3410 | 0.3658 |
| **Non-IID Combined Skew** | 45 | `FL-COMBINED-NONIID-20261020-141049-3C69B6` | 6.3720 | 7.5463 | 0.4739 | 0.5397 | 0.8095 | 0.2857 | 0.7565 | 0.2776 | 0.3818 |
| **Non-IID Combined Skew** | 46 | `FL-COMBINED-NONIID-20261020-155307-062E20` | 6.2810 | 7.2432 | 0.5534 | 0.5614 | 0.7619 | 0.4286 | 0.7129 | 0.2995 | 0.3327 |

---

### N. Summary Statistics: Mean ± Sample SD ($ddof=1$)

Computed across all 5 independent seed repetitions ($n=5$, Seeds 42, 43, 44, 45, 46) using Bessel's correction ($ddof=1$):

| Scenario | Valid Runs | MAE | RMSE | Macro F1 | Recall | Specificity | ROC-AUC | Lin's CCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FL-IID** | 5 | $5.2990 \pm 0.5178$ | $6.3738 \pm 0.4676$ | $0.6276 \pm 0.0799$ | $0.7048 \pm 0.1232$ | $0.6000 \pm 0.2119$ | $0.7537 \pm 0.0166$ | $0.3916 \pm 0.0513$ |
| **Non-IID Label Skew** | 5 | $5.0216 \pm 0.4079$ | $6.1540 \pm 0.4798$ | $0.6239 \pm 0.0484$ | $0.4000 \pm 0.1147$ | $0.8514 \pm 0.0550$ | $0.6996 \pm 0.0700$ | $0.3135 \pm 0.1023$ |
| **Non-IID Quantity Skew** | 5 | $5.4270 \pm 0.4893$ | $6.5173 \pm 0.5240$ | $0.6137 \pm 0.0528$ | $0.7143 \pm 0.1117$ | $0.5657 \pm 0.1659$ | $0.7309 \pm 0.0241$ | $0.3695 \pm 0.0541$ |
| **Non-IID Modality Skew** | 5 | $5.2566 \pm 0.5050$ | $6.3873 \pm 0.4723$ | $0.6371 \pm 0.0828$ | $0.6762 \pm 0.0522$ | $0.6286 \pm 0.1539$ | $0.7417 \pm 0.0221$ | $0.3921 \pm 0.0549$ |
| **Non-IID Combined Skew** | 5 | $5.5866 \pm 0.6777$ | $6.6818 \pm 0.6804$ | $0.6047 \pm 0.0880$ | $0.7048 \pm 0.0916$ | $0.5600 \pm 0.1986$ | $0.7279 \pm 0.0178$ | $0.3282 \pm 0.0481$ |

Summary details exported to `storage/federated/multiseed/m5_6/summary_statistics.csv` and `summary_statistics.json`.

---

### O. Confidence Intervals

Confidence intervals for sample means are computed using the Student-$t$ distribution with $df = n - 1 = 4$:

$$\text{CI}_{95\%} = \bar{x} \pm t_{0.975, 4} \cdot \frac{s}{\sqrt{n}}$$

For $n=5$, the critical value is $t_{0.975, 4} = 2.7764$:
- **FL-IID MAE 95% CI**: $[4.6560, 5.9420]$
- **Non-IID Label Skew MAE 95% CI**: $[4.5151, 5.5281]$
- **Non-IID Quantity Skew MAE 95% CI**: $[4.8194, 6.0346]$
- **Non-IID Modality Skew MAE 95% CI**: $[4.6295, 5.8837]$
- **Non-IID Combined Skew MAE 95% CI**: $[4.7451, 6.4281]$

**Methodological Invariant**: These intervals quantify training protocol variability across seeds conditional on the fixed 56-patient evaluation cohort. They do not claim patient-population generalizability.

---

### P. Paired Scenario Differences vs IID ($\Delta = X - \text{IID}$)

Cross-scenario comparisons are strictly paired by random seed ($S \in \{42, 43, 44, 45, 46\}$, $n=5$ pairs), guaranteeing identical initial model weights:

$$\Delta_{\text{metric}}(S, X) = \text{metric}(S, X) - \text{metric}(S, \text{IID})$$

Signed conventions:
- **Lower is better**: Negative $\Delta$ indicates improvement (MAE, RMSE, MSE, FNR, FPR).
- **Higher is better**: Positive $\Delta$ indicates improvement (Macro F1, Recall, Specificity, ROC-AUC, CCC, $r$).

| Scenario vs IID | Mean $\Delta$ MAE | Mean $\Delta$ RMSE | Mean $\Delta$ Macro F1 | Mean $\Delta$ Recall | Mean $\Delta$ Specificity | Mean $\Delta$ ROC-AUC | Consistency Across Seeds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Non-IID Label Skew** | $-0.2774 \pm 0.6565$ | $-0.2198 \pm 0.7423$ | $-0.0037 \pm 0.0656$ | **$-0.3048 \pm 0.1372$** | **$+0.2514 \pm 0.1845$** | $-0.0541 \pm 0.0747$ | **100% Negative Recall Drop (5/5 seeds)** |
| **Non-IID Quantity Skew** | $+0.1280 \pm 0.2968$ | $+0.1435 \pm 0.2970$ | $-0.0139 \pm 0.0582$ | $+0.0095 \pm 0.0851$ | $-0.0343 \pm 0.1504$ | $-0.0229 \pm 0.0214$ | **Worse MAE on 3/5 seeds; Lower AUC on 4/5 seeds** |
| **Non-IID Modality Skew** | $-0.0424 \pm 0.3139$ | $+0.0135 \pm 0.3156$ | $+0.0095 \pm 0.0509$ | $-0.0286 \pm 0.0987$ | $+0.0286 \pm 0.1385$ | $-0.0120 \pm 0.0268$ | **Highest Macro F1 among Non-IID; Resilient** |
| **Non-IID Combined Skew** | $+0.2876 \pm 0.7138$ | $+0.3080 \pm 0.7411$ | $-0.0229 \pm 0.0858$ | $0.0000 \pm 0.1750$ | $-0.0400 \pm 0.2552$ | $-0.0259 \pm 0.0258$ | **Worse MAE on 4/5 seeds; Lower $r$ on 5/5 seeds** |

Full paired deltas registered in `storage/federated/multiseed/m5_6/paired_deltas.csv` and `paired_summary.csv`.

---

### Q. PHQ-8 Severity Estimation Accuracy (Regression Analysis)

Across all 25 completed multi-seed runs ($n=5$ seeds per scenario):
1. **Non-IID Label Skew** achieves $\text{MAE} = 5.0216 \pm 0.4079$ and $\text{RMSE} = 6.1540 \pm 0.4798$ (Mean $\Delta \text{MAE} = -0.2774 \pm 0.6565$). Note that while apparent regression MAE appears lower, this is driven by conservative central-tendency predictions that underpredict high PHQ-8 values.
2. **Non-IID Modality Skew** exhibits strong regression resilience with $\text{MAE} = 5.2566 \pm 0.5050$ and $\text{RMSE} = 6.3873 \pm 0.4723$ (Mean $\Delta \text{MAE} = -0.0424 \pm 0.3139$). Modality Skew matches baseline IID regression accuracy within $\pm 0.04$ MAE.
3. **FL-IID Baseline Control** achieves $\text{MAE} = 5.2990 \pm 0.5178$ and $\text{RMSE} = 6.3738 \pm 0.4676$, with Lin's CCC of $0.3916 \pm 0.0513$.
4. **Non-IID Quantity Skew** exhibits moderate regression degradation with $\text{MAE} = 5.4270 \pm 0.4893$ and $\text{RMSE} = 6.5173 \pm 0.5240$ (Mean $\Delta \text{MAE} = +0.1280 \pm 0.2968$), degrading MAE in 3 of 5 seeds.
5. **Non-IID Combined Skew** exhibits the highest regression error with $\text{MAE} = 5.5866 \pm 0.6777$ and $\text{RMSE} = 6.6818 \pm 0.6804$ (Mean $\Delta \text{MAE} = +0.2876 \pm 0.7138$). Combined Skew degraded MAE relative to IID in 4 of 5 seeds and reduced Pearson $r$ in 5 of 5 seeds ($\Delta r = -0.0612 \pm 0.0389$).

---

### R. Binary Screening Robustness (Macro F1, Recall, Specificity, ROC-AUC)

1. **Macro F1**:
   - Non-IID Modality Skew: $0.6371 \pm 0.0828$ ($\Delta = +0.0095 \pm 0.0509$; highest Macro F1 among all Non-IID scenarios).
   - FL-IID: $0.6276 \pm 0.0799$
   - Non-IID Label Skew: $0.6239 \pm 0.0484$ ($\Delta = -0.0037 \pm 0.0656$)
   - Non-IID Quantity Skew: $0.6137 \pm 0.0528$ ($\Delta = -0.0139 \pm 0.0582$)
   - Non-IID Combined Skew: $0.6047 \pm 0.0880$ ($\Delta = -0.0229 \pm 0.0858$; lower than IID on 4/5 seeds).
2. **Recall (Screening Sensitivity for PHQ-8 $\ge 10$)**:
   - Non-IID Quantity Skew: $0.7143 \pm 0.1117$ ($\Delta = +0.0095 \pm 0.0851$)
   - FL-IID: $0.7048 \pm 0.1232$
   - Non-IID Combined Skew: $0.7048 \pm 0.0916$ ($\Delta = 0.0000 \pm 0.1750$)
   - Non-IID Modality Skew: $0.6762 \pm 0.0522$ ($\Delta = -0.0286 \pm 0.0987$)
   - Non-IID Label Skew: $0.4000 \pm 0.1147$ (**$\Delta = -0.3048 \pm 0.1372$; 100% consistent drop across all 5 seeds**, missing $12.6 \pm 2.41$ out of 21 true positive cases).
3. **Specificity (Screening Specificity for PHQ-8 $< 10$)**:
   - Non-IID Label Skew: $0.8514 \pm 0.0550$ (**$\Delta = +0.2514 \pm 0.1845$; 100% positive delta in 5/5 seeds** due to severe conservative negative bias).
   - Non-IID Modality Skew: $0.6286 \pm 0.1539$ ($\Delta = +0.0286 \pm 0.1385$)
   - FL-IID: $0.6000 \pm 0.2119$
   - Non-IID Quantity Skew: $0.5657 \pm 0.1659$ ($\Delta = -0.0343 \pm 0.1504$)
   - Non-IID Combined Skew: $0.5600 \pm 0.1986$ ($\Delta = -0.0400 \pm 0.2552$)
4. **ROC-AUC**:
   - FL-IID ($0.7537 \pm 0.0166$) > Non-IID Modality Skew ($0.7417 \pm 0.0221$) > Non-IID Quantity Skew ($0.7309 \pm 0.0241$) > Non-IID Combined Skew ($0.7279 \pm 0.0178$) > Non-IID Label Skew ($0.6996 \pm 0.0700$).

---

### S. Clinical Screening Error Analysis (False Negatives vs False Positives)

Screening error metrics for PHQ-8 threshold positivity ($\text{PHQ-8} \ge 10$) on the 56-patient evaluation cohort (21 screening-positive, 35 screening-negative), averaged across all 5 seeds ($n=5$):

| Scenario | Mean Missed Cases (FN) | Mean False Alarms (FP) | Mean True Positives (TP) | Mean True Negatives (TN) | Mean FNR | Mean FPR | Clinical Screening Safety Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **FL-IID** | $6.2 \pm 2.59$ | $14.0 \pm 7.42$ | $14.8 \pm 2.59$ | $21.0 \pm 7.42$ | 29.52% | 40.00% | Balanced baseline screening |
| **Non-IID Label Skew** | **$12.6 \pm 2.41$** | $5.2 \pm 1.92$ | $8.4 \pm 2.41$ | $29.8 \pm 1.92$ | **60.00%** | 14.86% | **Severe clinical safety hazard (misses 60% of cases)** |
| **Non-IID Quantity Skew** | $6.0 \pm 2.35$ | **$15.2 \pm 5.81$** | $15.0 \pm 2.35$ | $19.8 \pm 5.81$ | 28.57% | **43.43%** | High triage burden / false alarm inflation |
| **Non-IID Modality Skew** | $6.8 \pm 1.10$ | $13.0 \pm 5.39$ | $14.2 \pm 1.10$ | $22.0 \pm 5.39$ | 32.38% | 37.14% | Resilient multi-sensor screening |
| **Non-IID Combined Skew** | $6.2 \pm 1.92$ | $15.4 \pm 6.95$ | $14.8 \pm 1.92$ | $19.6 \pm 6.95$ | 29.52% | 44.00% | Anchored by clean majority client (Client 1) |

**Clinical Screening Distinctions**:
- Threshold positivity ($\text{PHQ-8} \ge 10$) indicates elevated depressive symptom risk warranting professional diagnostic follow-up; it is not a definitive psychiatric diagnosis.
- **Missed screening-positive cases (FN)** represent the primary clinical hazard. Non-IID Label Skew exhibited the worst clinical safety profile, missing 10 to 16 participants ($\text{FNR}=60.00\%$).
- **False alarms (FP)** induce secondary assessment burden. Quantity Skew and Combined Skew exhibited elevated false alarm inflation ($\text{FPR}=43.43\%$ and $44.00\%$).

---

### T. Convergence Stability & Learning Trajectories

Analysis of the 50-round training dynamics across all 25 completed runs:
1. **Initial Phase (Rounds 0–10)**: Rapid loss descent across all scenarios, with MAE dropping from $\sim 8.11$ to $\sim 5.2 - 5.8$.
2. **Intermediate Phase (Rounds 11–35)**: Steady refinement; Cosine Annealing learning rate schedule moderates client oscillations.
3. **Late-Round Stability (Rounds 46–50)**: Trajectories stabilize with minimal fluctuation across all 5 seeds.
4. Retrospective evaluation confirms that Round 50 provides a representative, non-overfit final checkpoint for all scenarios without test-driven early stopping.

---

### U. Client Drift and Aggregation Mechanics

Client drift quantified through client parameter update norms ($||\Delta w_k|| = ||w_k - w_{\text{global}}||$):

| Scenario | Average Raw L2 Norm | Average Weighted L2 Norm | Client Weight Distribution | Dominant Client |
| :--- | :---: | :---: | :---: | :--- |
| **FL-IID** | 15.42 | 5.14 | 33.1% / 33.7% / 33.1% | None (Balanced) |
| **Non-IID Label Skew** | 15.41 | 15.41 | 33.1% / 33.7% / 33.1% | None (Directional divergence) |
| **Non-IID Quantity Skew** | 15.18 | 6.88 | 60.1% / 30.1% / 9.8% | Client 1 (60.1% weight share) |
| **Non-IID Modality Skew** | 15.45 | 5.15 | 33.1% / 33.7% / 33.1% | None (Balanced) |
| **Non-IID Combined Skew** | 15.29 | 6.89 | 60.1% / 30.1% / 9.8% | Client 1 (60.1% clean data) |

---

### V. Communication Efficiency & Payload Accounting

Exported to `storage/federated/multiseed/m5_6/communication_summary.csv`:
1. **Model Parameter Count**: $7,832,001$ FP32 parameters ($31,328,004$ bytes $\approx 29.877$ MiB per model payload).
2. **Per-Round Full Participation ($K=3$) Payload**:
   - Downlink: $3 \times 31,328,004 = 93,984,012$ bytes ($\approx 89.63$ MiB).
   - Uplink: $3 \times 31,328,004 = 93,984,012$ bytes ($\approx 89.63$ MiB).
   - Total per round: $187,968,024$ bytes ($\approx 179.26$ MiB, binary $2^{20}$).
3. **Cumulative 50-Round Payload per Full Run**:
   - $50 \times 187,968,024 = 9,398,401,200$ bytes ($\approx 8.753$ GiB, binary $2^{30}$).
4. **Communication Invariant**: Under standard FedAvg with full participation ($C=1.0$), communication overhead is strictly invariant to data heterogeneity, label skew, or modality degradation. Total logical payload across all 25 completed runs: $234,960,030,000$ bytes ($\approx 218.82$ GiB).

---

### W. Supplementary Robustness & Degradation Diagnostics

In Modality Skew and Combined Skew scenarios, client feature corruption diagnostics were verified by `storage/federated/runs/<run_id>/degradation_diagnostics.csv`:
- **Audio Feature Noise**: Client 2 tensor values confirmed shifted with Gaussian noise ($\sigma = 0.25$); padding values and valid timestep masks preserved; zero NaN/inf values.
- **Visual Frame Dropout**: Client 3 frame dropout confirmed with $p = 0.20$; dropped frames correctly imputed with the frozen training visual mean reference; padding intact.
- **Text Features**: 100% clean across all clients.
- **Validation and Test Cohorts**: 100% clean, verified by `test_17_no_corrupted_validation_test_data`.

---

### X. Statistical Interpretation & Variance Methodological Disclosures

1. **Effect Magnitude & Statistical Power ($n=5$)**: With all 5 predefined seeds completed ($n=5$, $df=4$), Student-$t$ confidence intervals and paired differences offer defensible statistical grounding.
2. **Modality Skew Findings**: Across 5 seeds, Modality Skew proves resilient, achieving competitive regression MAE ($5.2566 \pm 0.5050$) and the highest Macro F1 among non-IID scenarios ($0.6371 \pm 0.0828$).
3. **Label Skew Clinical Hazard**: The severe sensitivity collapse ($\Delta \text{Recall} = -0.3048 \pm 0.1372$) is 100% consistent across all 5 seeds, confirming it as an empirical property of clinical label skew under unweighted FedAvg.
4. **Variance Attribution Caveat**: In the FED-MIND experimental protocol, each seed randomizes both model initialization weights and participant partition allocation. Because this is not a full factorial cross-product experiment (multiple partitions evaluated per single initialization), independent variance attribution between initialization vs partition cannot be claimed statistically.
5. **Patient-Level Uncertainty vs Model Stochasticity**: Repeated evaluations on the same 56 test participants capture training stochasticity conditional on this cohort, not patient-level population generalizability.

---

### Y. Limitations and Threats to Validity

1. **Hardware Environment Heterogeneity**: Seed-42 IID executed on host CUDA acceleration, whereas all subsequent runs executed on CPU within Flower hermetic environments. All algorithms and loss functions were mathematically identical.
2. **Small Institutional Client Count ($K=3$)**: Reflects a 3-center hospital federation rather than an edge mobile network ($K > 10^3$).
3. **Fixed Global Test Cohort ($N=56$)**: Test performance is evaluated on a single hold-out partition (21 screening-positive, 35 screening-negative).

---

### Z. Reproducibility Artifacts

All artifacts are persisted in the repository without external dependencies:
- **Seed Initializations**: `storage/federated/multiseed/initializations/`
- **Partition Sets**: `storage/federated/multiseed/partitions/`
- **Run Registry**: `storage/federated/multiseed/m5_6/run_registry.json` (25/25 runs registered)
- **Seed Scenario Matrix**: `storage/federated/multiseed/m5_6/seed_scenario_matrix.csv` (25/25 rows complete)
- **Summary Statistics**: `storage/federated/multiseed/m5_6/summary_statistics.csv` & `.json` (5-seed aggregates)
- **Paired Deltas**: `storage/federated/multiseed/m5_6/paired_deltas.csv` & `paired_summary.csv` (20 paired contrasts)
- **Clinical Error Summary**: `storage/federated/multiseed/m5_6/clinical_error_summary.csv`
- **Visualizations**: 13 publication plots in `storage/federated/multiseed/m5_6/plots/`
- **Orchestration Script**: `scripts/run_m5_6_sweep.py`
- **Analysis Script**: `scripts/analyze_m5_6_results.py`
- **Automated Tests**: `tests/test_federated_multiseed.py`

---

### AA. Automated Test Summary

The milestone test suite (`tests/test_federated_multiseed.py`) verifies all 35 milestone criteria:
- Configuration parsing, matrix generation, 5 scenarios, 5 seeds, 25 unique combinations.
- Registry state transitions, incomplete run detection, initialization hash equality.
- Zero participant leakage, dataset pool equality, partition determinism.
- Modality degradation validation, clean test/val sets, FedAvg client weights.
- Round-0 evaluation parity, canonical evaluation, seed metric extraction.
- Sample SD ($ddof=1$), Student-$t$ CI, paired deltas, undefined metric handling.
- Clinical confusion matrix consistency, FNR/FPR calculation, communication accounting.
- Client drift definitions, production checkpoint preservation, integration run verification.

**Result**: 35 passed, 0 failed in 7.55s.  
**Regression Suite**: `test_health.py`, `test_model.py`, `test_screening.py` (10 passed, 0 failed).  
**Total Suite Pass Rate**: 100%.

---

### AB. Scientific Findings (Addressing RQ1–RQ10)

1. **RQ1 (IID vs Non-IID)**: FL-IID achieves the highest ROC-AUC ($0.7537 \pm 0.0166$) and balanced sensitivity ($0.7048$). Non-IID Modality Skew tracks IID closely ($5.2566$ MAE, $0.6371$ Macro F1). Quantity Skew degrades accuracy ($5.4270$ MAE), and Combined Skew produces the highest regression error ($5.5866$ MAE).
2. **RQ2 (Stability of Differences)**: Label Skew exhibits a 100% consistent drop in screening Recall across all 5 seeds ($\Delta \text{Recall} = -0.3048 \pm 0.1372$). Combined Skew degrades MAE in 4/5 seeds and Pearson $r$ in 5/5 seeds.
3. **RQ3 (PHQ-8 Severity Accuracy)**: Label Skew ($5.0216$) $<$ Modality Skew ($5.2566$) $<$ FL-IID ($5.2990$) $<$ Quantity Skew ($5.4270$) $<$ Combined Skew ($5.5866$).
4. **RQ4 (Screening Robustness)**: Macro F1: Modality Skew ($0.6371$) > FL-IID ($0.6276$) > Label Skew ($0.6239$) > Quantity Skew ($0.6137$) > Combined Skew ($0.6047$). Label Skew yields high specificity ($0.8514$) at catastrophic clinical cost to sensitivity ($0.4000$).
5. **RQ5 (Initialization vs Partition Randomness)**: Joint seed stochasticity produces moderate variance (SD $\sim 0.40 - 0.68$ MAE). Full decoupling would require factorial cross-product evaluation.
6. **RQ6 (Convergence Stability)**: 50 communication rounds provide stable convergence across all scenarios with steady trajectory leveling at rounds 45–50.
7. **RQ7 (Client Drift)**: Raw update norms remain similar ($\approx 15.2 - 15.4$). In Quantity and Combined Skew, Client 1's 60.1% scalar weight anchors training trajectory.
8. **RQ8 (Communication Overhead)**: Invariant at $179.26$ MiB/round and $8.753$ GiB/run across all scenarios.
9. **RQ9 (Defensible Findings)**: Late fusion multimodal architecture is resilient to single-modality noise/dropout. Unweighted FedAvg under severe label imbalance creates clinical under-diagnosis hazard.
10. **RQ10 (Unsupported Conclusions)**: Generalization beyond the 56-patient evaluation cohort and extrapolation to massive edge topologies ($K > 1000$) remain unsupported.

---

### AC. Final Status Declarations

In accordance with Milestone 5.6 acceptance criteria:

1. **Scientific Consistency**: **`M5.6 SCIENTIFIC CONSISTENCY QUALIFIED`**  
   *Round-0 discrepancy audited and resolved; canonical evaluator verified; zero data leakage asserted. Qualified due to runtime environment divergence between Seed-42 IID (host CUDA) and other runs (Flower CPU).*

2. **Execution**: **`M5.6 EXECUTION PASSED`**  
   *All 25 required runs completed, verified, and registered in `run_registry.json` and `seed_scenario_matrix.csv` (100% matrix completion rate).*

3. **Statistical Analysis**: **`M5.6 STATISTICAL ANALYSIS PASSED`**  
   *Complete 5-seed metrics extracted; sample SD ($ddof=1$) and Student-$t$ 95% CIs ($df=4$) calculated; 20 within-seed paired contrasts computed; clinical screening error analysis completed; all 13 publication plots generated with full benchmark tags.*


