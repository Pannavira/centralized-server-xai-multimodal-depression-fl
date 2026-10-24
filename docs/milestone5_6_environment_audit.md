# FED-MIND — Milestone 5.6 Environment Consistency Audit
**Computational Infrastructure, Framework Stack, and Numerical Precision Parity Audit**  
**Date:** 2026-10-20  
**Status:** **AUDITED AND FROZEN**

---

## 1. Executive Summary

Milestone 5.6 establishes a cross-scenario, multi-seed statistical evaluation requiring strict software and hardware transparency. This audit details the active execution stack, documents differences across historical runs, and establishes the frozen environment configuration for all Milestone 5.6 executions.

---

## 2. Comprehensive System Manifest

| Component | Specification / Version | Verification Method |
| :--- | :--- | :--- |
| **Operating System** | Microsoft Windows 11 Home (Build 10.0.26200-SP0) | `platform.platform()` |
| **Host CPU** | x86_64 AMD64 Architecture (12 Logical Cores) | System inspection |
| **Host GPU** | NVIDIA GeForce RTX 3050 6GB Laptop GPU | `nvidia-smi` |
| **GPU VRAM** | 6144 MiB Dedicated GDDR6 | `torch.cuda.get_device_properties()` |
| **NVIDIA Driver** | 591.74 | `nvidia-smi` |
| **CUDA Driver API** | 13.1 | `nvidia-smi` |
| **CUDA Runtime API**| 12.8 | `torch.version.cuda` |
| **cuDNN Library** | 91900 | `torch.backends.cudnn.version()` |
| **Python Runtime** | 3.14.3 (tags/v3.14.3:323c59a, Feb 3 2026) | `sys.version` |
| **PyTorch (Venv)** | 2.11.0+cu128 | `torch.__version__` |
| **PyTorch (Flower)**| 2.14.1+cpu | `torch.__version__` |
| **Flower Framework**| 1.40.0 (Message API, ServerApp, ClientApp) | `flwr.__version__` |
| **Ray Core** | 2.59.0 | `ray.__version__` |
| **NumPy** | 2.4.4 (Venv) / 2.5.4 (Flower) | `numpy.__version__` |
| **SciPy** | 1.17.1 (Venv) / 1.18.1 (Flower) | `scipy.__version__` |
| **Scikit-Learn** | 1.8.0 (Venv) / 1.9.1 (Flower) | `sklearn.__version__` |
| **Pandas** | 3.0.2 (Venv) / 3.0.6 (Flower) | `pandas.__version__` |
| **Git Commit** | `828c079d5c4ecffb935d77d93e9d48a9f683d612` | `git rev-parse HEAD` |
| **Lockfile** | `uv.lock` / `pyproject.toml` | Workspace root |

---

## 3. Historical Runs Environment Parity Audit

| Run Identifier | Milestone & Scenario | Python | PyTorch | Device | CUDA Version | Flower | Ray |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `FL-RUN-20261016-130403-45D587` | M5.1 FL-IID Control | 3.14.3 | 2.11.0+cu128 | CUDA (RTX 3050) | 12.8 | 1.40.0 | 2.59.0 |
| `FL-LABEL-SKEW-20261016-141108-7865D4` | M5.2 Non-IID Label Skew | 3.14.3 | 2.14.1+cpu | CPU | None | 1.40.0 | 2.59.0 |
| `FL-QUANTITY-SKEW-20261019-110746-3A4BFF` | M5.3 Non-IID Quantity Skew | 3.14.3 | 2.14.1+cpu | CPU | None | 1.40.0 | 2.59.0 |
| `FL-MODALITY-SKEW-20261019-142242-CA6D11` | M5.4 Non-IID Modality Skew | 3.14.3 | 2.14.1+cpu | CPU | None | 1.40.0 | 2.59.0 |
| `FL-COMBINED-NONIID-20261019-154311-CCF7B4`| M5.5 Combined Non-IID | 3.14.3 | 2.14.1+cpu | CPU | None | 1.40.0 | 2.59.0 |

### 3.1 Evaluation of Hardware Precision Differences (CUDA vs CPU)
In M5.1, local training occurred with GPU acceleration (`2.11.0+cu128`), whereas M5.2–M5.5 executed under Flower's simulation workers with CPU tensor execution (`2.14.1+cpu`).

Forensic comparison demonstrates:
1. **Initial Checkpoints (Round 0)**:
   - Checkpoint bitwise hash is identical across all five runs: `a1c052476e...`.
   - Continuous prediction differences on hold-out test set are strictly $\mathbf{0.00000000}$.
2. **Final Checkpoints (Round 50)**:
   - Re-evaluating M5.1 Round-50 checkpoint on CPU produces $\text{MAE} = 4.9997$ and $\text{RMSE} = 6.1402$, identical down to 4 decimal places with the recorded metrics in `metrics.csv`.
   - Across device architectures, cuBLAS vs OpenBLAS/oneDNN matrix multiplications differ by at most $\sim 5 \times 10^{-4}$ due to floating-point summation order (non-associativity).
3. **Verdict on Historical Inclusions**:
   - The five seed-42 runs exhibit genuine scientific reproducibility and represent valid empirical observations.
   - They are incorporated into the multi-seed registry with the explicit qualification `REUSED_VERIFIED`.

---

## 4. Frozen Execution Environment for Milestone 5.6

To ensure absolute comparability across all 25 cells of the experiment matrix:
1. All newly executed runs (seeds 43, 44, 45, 46 across all scenarios) will use the stable Flower 1.40.0 execution runtime.
2. Canonical evaluation for all runs (both historical seed 42 and newly trained seeds) will be conducted using `GlobalEvaluator` executed on CPU with deterministic PyTorch configurations.
3. No framework or major library upgrades will take place during Milestone 5.6.
