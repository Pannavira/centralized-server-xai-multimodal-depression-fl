# FED-MIND — Milestone 5.6 Seed & Reproducibility Protocol
**Deterministic Seed Hierarchy, Random Number Generator Decoupling, and Pairing Governance**  
**Date:** 2026-10-20  
**Status:** **FROZEN PROTOCOL**

---

## 1. Executive Summary

Milestone 5.6 validates five federated learning scenarios across five independent random seeds ($S \in \{42, 43, 44, 45, 46\}$). To maintain strict paired comparability while preventing cross-scenario state pollution, this protocol governs the hierarchical derivation of pseudo-random number generator (RNG) states across all experimental dimensions.

---

## 2. Seed Hierarchy Architecture

To ensure independence between model initialization, dataset partitioning, local stochastic training, and modality corruption, a structured RNG hierarchy is enforced:

```text
Base Seed S ∈ {42, 43, 44, 45, 46}
│
├── 1. Model Initialization Seed : S_model = S
│      └── Initial weights frozen at Round 000 (shared across all 5 scenarios for seed S)
│
├── 2. Partition Allocation Seed  : S_part = S
│      └── NumPy RandomState(S) permutes positive/negative strata for client assignment
│
├── 3. Training & DataLoader Seed : S_train = (S * 1000) + (round_num * 10) + client_id
│      └── Local mini-batch shuffling and optimizer stochasticity
│
├── 4. Modality Corruption Seed   : S_degrade = S
│      └── Client 2 Gaussian audio noise (σ=0.25) & Client 3 visual frame dropout (p=0.20)
│
└── 5. Canonical Evaluation Seed  : Deterministic Fixed Order
       └── Test DataLoader has shuffle=False, drop_last=False, identical sample ordering
```

---

## 3. Seed Allocation & Pairing Matrix

Within each seed row $S$, **all five scenarios share the exact same Round-0 checkpoint**:

| Seed ($S$) | Initial Checkpoint Source | Initial Checkpoint SHA-256 | Scenario 1: IID | Scenario 2: Label Skew | Scenario 3: Quantity Skew | Scenario 4: Modality Skew | Scenario 5: Combined Skew |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | Historical M5.1 R0 | `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973` | REUSED_VERIFIED | REUSED_VERIFIED | REUSED_VERIFIED | REUSED_VERIFIED | REUSED_VERIFIED |
| **43** | Fresh initialization | Seed 43 bitwise frozen | NEW_RUN | NEW_RUN | NEW_RUN | NEW_RUN | NEW_RUN |
| **44** | Fresh initialization | Seed 44 bitwise frozen | NEW_RUN | NEW_RUN | NEW_RUN | NEW_RUN | NEW_RUN |
| **45** | Fresh initialization | Seed 45 bitwise frozen | NEW_RUN | NEW_RUN | NEW_RUN | NEW_RUN | NEW_RUN |
| **46** | Fresh initialization | Seed 46 bitwise frozen | NEW_RUN | NEW_RUN | NEW_RUN | NEW_RUN | NEW_RUN |

---

## 4. Determinism and Independence Invariants

1. **Paired Seed Equality**:
   For any seed $S$, the Round-0 checkpoint SHA-256 must match exactly across all five scenarios:
   $$\text{SHA256}(w_{0, \text{IID}}^{(S)}) \equiv \text{SHA256}(w_{0, \text{Label}}^{(S)}) \equiv \text{SHA256}(w_{0, \text{Quantity}}^{(S)}) \equiv \text{SHA256}(w_{0, \text{Modality}}^{(S)}) \equiv \text{SHA256}(w_{0, \text{Combined}}^{(S)})$$
2. **Across-Seed Independence**:
   Different seeds must produce genuinely distinct initial parameter configurations:
   $$\text{SHA256}(w_{0}^{(S_i)}) \ne \text{SHA256}(w_{0}^{(S_j)}) \quad \forall i \ne j$$
3. **Execution Order Decoupling**:
   Scenario execution order (e.g. running IID before Label Skew vs running Label Skew before IID) must have zero effect on RNG state. Each run instantiates isolated RNGs derived exclusively from explicit seed values, never relying on process-level global mutable state.
4. **No Process-Randomized Python `hash()`**:
   No partition allocation or feature transform uses Python's built-in `hash()`, which is randomized across process restarts by Python's hash seed randomization. All hashes use SHA-256.
5. **No Seed Reset to 42 During Local Training**:
   Local training workers receive their seed dynamically derived from $(S, \text{round}, \text{client})$, preventing artificial weight convergence across different seeds.
