# Milestone 5.1: Audit Historis Eksperimen Federated Learning (FL-IID)

**Dokumen**: `docs/milestone5_1_historical_audit.md`  
**Proyek**: FED-MIND (Centralized Server & Federated Learning Infrastructure)  
**Waktu Audit**: 2026-10-16  
**Repositori Referensi**: `C:\Wira\Kuliah\Penelitian\clinical-xai-multimodal-depression-fl`  
**Status Audit**: SELESAI & DIVERIFIKASI FAKTUAL  

---

## 1. Ringkasan Eksekutif & Klasifikasi Reproduksi

Audit komprehensif dilakukan langsung terhadap source code (`federated_learning/`), file konfigurasi YAML (`federated_training_config.yaml`, `federated_training_config_v2.yaml`), checkpoint fisik `.pt`, dan log eksekusi historis (`federated_learning_output/experiments/` dan `experiments_v2/`).

### Klasifikasi Ilmiah:
```text
scientific_status = "research_reproduction_fl_iid_approximate"
```
**Justifikasi**:
1. Seluruh hiperparameter pelatihan lokal, partisi dataset, loss function, arsitektur neural network, optimizer, scheduler, dan random seed terverifikasi secara persis (100% cocok).
2. Terdapat perbedaan faktual antara **desain protokol** (50 ronde di `federated_training_config.yaml`) dan **eksekusi fisik historis**:
   - `fedavg_iid_3clients_randominit` (v1) hanya dieksekusi selama **12 ronde** (berhenti di ronde 12 dengan model terbaik terpilih di Ronde 7).
   - `fedavg_iid_3clients_pretrained` (v2) dieksekusi selama **20 ronde** (warm-start dari `modeling2/output/best_model.pt`).
3. Ditemukan **cacat metodologis pada pipeline historis** di mana pemilihan model terbaik (`best_global_model.pt`) dilakukan berdasarkan skor metrik pada `global_test_index.csv`. Pada Milestone 5.1 ini, test set ditegakkan secara ketat sebagai **hold-out reporting only** tanpa campur tangan model selection.
4. Oleh karena itu, eksperimen 50 ronde FL-IID dari inisialisasi deterministik seed 42 diklasifikasikan sebagai **reproduksi aproksimasi metodologis yang valid secara ilmiah**, bukan replikasi bitwise semu.

---

## 2. Matriks Parameter Historis Terverifikasi

| Parameter | Konfigurasi Historis v1 (`experiments/`) | Konfigurasi Historis v2 (`experiments_v2/`) | Desain Protokol YAML | Implementasi M5.1 | Bukti File Referensi |
|:---|:---:|:---:|:---:|:---:|:---|
| **Jumlah Klien** | 3 | 3 | 3 | **3** | `federated_training_config.yaml:L84` |
| **Partisipasi Klien** | 100% (3/3) | 100% (3/3) | 1.0 | **1.0 (3/3)** | `federated_training_config.yaml:L85` |
| **Algoritma Agregasi** | FedAvg terbobot | FedAvg terbobot | FedAvg | **FedAvg (`num-examples`)** | `simulation.py:L76`, `client.py:L218` |
| **Jumlah Ronde** | 12 ronde (Best R7) | 20 ronde (Best R0/R7) | 50 ronde | **50 ronde** | `fl_experiment_summary.json` vs YAML |
| **Local Epochs** | 2 | 2 | 2 | **2** | `federated_training_config.yaml:L93` |
| **Batch Size** | 16 | 16 | 16 | **16** | `federated_training_config.yaml:L106` |
| **Optimizer** | AdamW | AdamW | AdamW | **AdamW** | `client.py:L148` |
| **Learning Rate** | 1e-4 (`0.0001`) | 5e-5 (`0.00005`) | 1e-4 | **1e-4 (`0.0001`)** | `federated_training_config.yaml:L107` |
| **Weight Decay** | 0.01 | 0.0001 | 0.01 | **0.01** | `federated_training_config.yaml:L108` |
| **LR Scheduler** | `CosineAnnealingLR` | `CosineAnnealingLR` | `cosine_annealing` | **`CosineAnnealingLR`** | `client.py:L149` |
| **Siklus Scheduler** | Reset tiap ronde lokal | Reset tiap ronde lokal | Reset per fit | **Reset tiap ronde lokal** | `client.py:L148-L149` (`T_max=epochs`) |
| **Fungsi Loss** | `ClassWeightedMSELoss` | `ClassWeightedMSELoss` | `class_weighted_mse` | **`ClassWeightedMSELoss`** | `loss.py:L18`, `client.py:L89` |
| **Pos Weight Multiplier** | 2.5 | 1.5 | 2.5 | **2.5** | `federated_training_config.yaml:L114` |
| **False Negative Penalty** | 2.0 | 1.2 | 2.0 | **2.0** | `federated_training_config.yaml:L115` |
| **Clinical Cutoff** | PHQ-8 $\ge 10.0$ | PHQ-8 $\ge 10.0$ | 10.0 | **10.0** | `federated_training_config.yaml:L29` |
| **Gradient Clipping** | 1.0 (`clip_grad_norm_`) | 1.0 | 1.0 | **1.0** | `client.py:L178` |
| **Random Seed** | 42 | 42 | 42 | **42** | `federated_training_config.yaml:L118` |
| **Inisialisasi Model** | Deterministic Seed 42 RandomInit | Pretrained (`modeling2`) | RandomInit (default) | **Seed 42 RandomInit** | `simulation.py:L182-L210` |
| **Early Stopping** | Nonaktif (`false`) | Nonaktif (`false`) | false | **false** | `federated_training_config.yaml:L117` |
| **Dropout Encoders** | 0.4 (Conv) | 0.4 (Conv) | 0.4 | **0.4** | `models.py` |
| **Dropout Regressor** | 0.3 | 0.3 | 0.3 | **0.3** | `models.py` |
| **Arsitektur Model** | `MultimodalBaselineModel` | `MultimodalBaselineModel` | Identik | **`MultimodalBaselineModel`** | `app/core/model.py` |

---

## 3. Audit Partisi Data & Integritas Privasi

### A. Partisi Klien Resmi
- **Lokasi Partisi**: `C:\Wira\Kuliah\Penelitian\clinical-xai-multimodal-depression-fl\federated_learning_output\IID_3Clients\`
  1. `client_1_train.csv`: 54 partisipan
  2. `client_1_val.csv`: 14 partisipan
  3. `client_2_train.csv`: 55 partisipan
  4. `client_2_val.csv`: 14 partisipan
  5. `client_3_train.csv`: 54 partisipan
  6. `client_3_val.csv`: 13 partisipan
- **Total Sampel Latih**: 163 partisipan (100% data latih E-DAIC terbagi rata terstratifikasi).
- **Total Sampel Validasi**: 41 partisipan.

### B. Global Hold-Out Test Set
- **Lokasi File**: `C:\Wira\Kuliah\Penelitian\clinical-xai-multimodal-depression-fl\federated_learning_output\global_test_index.csv`
- **Jumlah Sampel**: 56 partisipan unik.

### C. Pembuktian Zero Data Leakage
$$\text{intersection}(\text{all\_client\_train\_ids}, \text{global\_test\_ids}) = \emptyset$$
$$\text{intersection}(\text{all\_client\_val\_ids}, \text{global\_test\_ids}) = \emptyset$$
- Hasil pengujian: **0 overlap subjek** (Terkonfirmasi Zero Leakage 100%).

---

## 4. Temuan Kritis: Checkpoint Referensi & Model Selection

### A. Checkpoint Produksi yang Telah Terpasang di Prototype
- Lokasi: `storage/models/fl_iid/model.pt`
- File Size: `31,355,213 bytes`
- **SHA-256**: `17d59aa90455386a1e2f00bed2107ef0cbe2a6fb3e96e3f476faf0272db0ec73`
- Asal Checkpoint: Identik 100% bitwise dengan checkpoint hasil eksperimen v2 `experiments_v2/fedavg_iid_3clients_pretrained/best_global_model.pt`.
- Model ini berasal dari inisialisasi warm-start centralized baseline (`modeling2/output/best_model.pt`) yang dilatih selama 20 ronde.

### B. Checkpoint Eksperimen RandomInit (v1)
- Lokasi: `experiments/fedavg_iid_3clients_randominit/best_global_model.pt`
- File Size: `32,938,657 bytes`
- **SHA-256**: `b839ad9f5537036e028f9f0508b6cb91f7785203911898b9cb15a9dd11f0921c`
- Metrik Referensi Historis (Ronde 7 / 12):
  - **F1-Score**: 0.4932
  - **Recall**: 1.0000
  - **Precision**: 0.3273
  - **MAE**: 6.8182
  - **MSE**: 61.4867
  - **CCC**: -0.0254
  - **Test Loss**: 42.9196

### C. Dekoupling Evaluasi Global vs Model Selection
- **Praktik Historis**: `strategy.py` secara langsung memeriksa `if current_f1 > self.best_f1` pada global test set.
- **Standar Protokol M5.1**:
  - Global hold-out test set ($N=56$) hanya dievaluasi tiap ronde untuk tujuan pelaporan longitudinal (`metrics.csv`).
  - Model selection hanya dapat didasarkan pada metrik validasi lokal klien teragregasi (`avg_client_val_loss` / `avg_client_val_mae`) atau model final Round 50 (`final_round_050/model.pt`).
  - Tidak ada pemilihan model berbasis test set.

---

## 5. Parameter yang Tidak Diketahui / Nondeterministik
- **Status Parameter**: Seluruh parameter arsitektur, data, optimasi, dan loss telah berhasil diidentifikasi.
- **Sumber Nondeterminisme**: Operasi atomik CUDA (misalnya `torch.bmm`, `scatter_add`, atomic reduction) menghasilkan variasi floating-point kecil antar eksekusi GPU. Oleh karena itu, reproduksi dinilai berdasarkan konvergensi metrik klinis dalam rentang toleransi ilmiah yang wajar, bukan bitwise hash matching terhadap run masa lalu.
