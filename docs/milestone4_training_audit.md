# Pre-Flight & Training Objective Audit: Milestone 4 (Federated Learning Infrastructure)

**Tanggal Audit:** 16 Oktober 2026  
**Repositori Target:** `centralized-server-xai-multimodal-depression-fl`  
**Repositori Referensi Riset:** `C:\Wira\Kuliah\Penelitian\clinical-xai-multimodal-depression-fl`  
**Status:** Terverifikasi Lengkap

---

## 1. Lingkungan Komputasi & Dependensi

| Komponen | Spesifikasi / Nilai Terverifikasi | Status |
| :--- | :--- | :--- |
| **Python Version** | `3.14.3` (tags/v3.14.3, 64-bit AMD64) | OK |
| **PyTorch Version** | `2.11.0+cu128` | OK |
| **CUDA Acceleration** | `True` (Tersedia) | OK |
| **GPU Hardware** | `NVIDIA GeForce RTX 3050 6GB Laptop GPU` (VRAM: 6.44 GB) | OK |
| **Flower Framework** | `flwr[simulation]==1.40.0` (Terinstal & Terverifikasi) | OK |
| **Pytest Baseline** | 10 passed, 0 failed, 1 warning (3.49s) | OK |
| **Git Working Tree** | Clean (`main` branch, commit `828c079`) | OK |

---

## 2. Audit Arsitektur Model Multimodal

- **Authoritative Model Class:** `MultimodalBaselineModel` ([app/core/model.py](file:///c:/Wira/Kuliah/Penelitian/PROTOTYPE%20APLIKASI%20FL/centralized-server-xai-multimodal-depression-fl/app/core/model.py))
- **Kompatibilitas:** Identik 100% dengan `clinical-xai-multimodal-depression-fl/federated_learning/Model_Architecture/models.py`.
- **Dimensi Input Sensor:**
  - Audio: `(B, 512, 80)` — Log-Mel Spectrogram (80 mel filterbanks)
  - Visual: `(B, 512, 72, 3)` — 68 3D Facial Landmarks + 4 Gaze Vectors
  - Text: `(B, 512, 728)` — Sentence-BERT Embeddings
- **Encoder:** Masing-masing modalitas menggunakan `Conv1d`/`Conv2d` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `BiLSTM` (hidden=256) $\rightarrow$ `TemporalAttentionPooling`.
- **Fusion:** Late Fusion Concatenation ($256 \times 3 = 768$) $\rightarrow$ Linear Regressor Head $\rightarrow$ Skor PHQ-8 kontinu (dimensi output: 1).
- **Total Parameter:** 7.832.001 parameter.

---

## 3. Audit Objektif Pelatihan & Diskrepansi Metodologis

### A. Pertanyaan Inti: Apakah Pelatihan Menggunakan BCE atau Regresi PHQ-8?
- **Temuan Dokumen Proposal/Protokol:** Beberapa dokumen awal mencatat "Adam, learning rate 1e-4, BCE Loss".
- **Temuan Faktual pada Kode Riset Asli:**
  Pada implementasi resmi `C:\Wira\Kuliah\Penelitian\clinical-xai-multimodal-depression-fl\`:
  - File konfigurasi: `federated_training_config.yaml`
  - Modul loss: `Model_Architecture/loss.py`
  - Kelas loss default: **`ClassWeightedMSELoss`**
  - Parameter:
    - `binarize_threshold: 10.0` (ambang batas klinis depresi)
    - `pos_weight_multiplier: 2.5` (bobot ekstra untuk sampel $\ge 10.0$)
    - `fn_penalty: 2.0` (penalti untuk prediksi di bawah 10.0 pada kasus depresi)
  - Target output model: **Skor Kontinu PHQ-8** (`float` antara 0.0 s.d 24.0), **BUKAN output probabilitas biner tunggal sigmoid/BCE**.
- **Kesimpulan Audit:** Target objektif FL sebenarnya adalah **Clinical Weighted PHQ-8 Regression**. Protokol klasifikasi biner dievaluasi secara downstream menggunakan ambang batas $\text{PHQ-8} \ge 10.0$.

### B. Hyperparameter Pelatihan Lokal Klien (Official Baseline)
- **Optimizer:** `AdamW`
- **Learning Rate:** `1e-4` (`0.0001`)
- **Weight Decay:** `0.01`
- **Scheduler:** `CosineAnnealingLR` ($T_{\max}=\text{local\_epochs}$, $\eta_{\min}=10^{-7}$)
- **Batch Size:** `16`
- **Local Epochs (Riset):** `2`
- **Local Epochs (Smoke Test M4):** `1` (sesuai instruksi uji infrastruktur awal)
- **Random Seed:** `42`

### C. Inisialisasi Model Global
- Dokumen riset menetapkan perbandingan adil FL vs Centralized menggunakan **Deterministic Seed-42 Random Initialization** (`initialize_from_centralized_baseline: false`).
- Untuk Milestone 4 Infrastructure Smoke Test: Model awal diinisialisasi secara deterministik dengan seed 42 dan diberi label metadata `scientific_status="infrastructure_smoke_test_not_research_result"`. File produksi `storage/models/fl_iid/model.pt` tidak boleh ditimpa.

---

## 4. Partisi Dataset Klien & Verifikasi Zero-Leakage

### A. Lokasi File Partisi Resmi
- Direktori Partisi IID (3 Klien):
  `C:\Wira\Kuliah\Penelitian\clinical-xai-multimodal-depression-fl\federated_learning_output\IID_3Clients\`
  - `client_1_train.csv` (54 sampel train)
  - `client_1_val.csv` (56 sampel val)
  - `client_2_train.csv` (55 sampel train)
  - `client_2_val.csv` (56 sampel val)
  - `client_3_train.csv` (54 sampel train)
  - `client_3_val.csv` (56 sampel val)
- Dataset Fitur Multimodal:
  `C:\Wira\Kuliah\Penelitian\clinical-xai-multimodal-depression-fl\modeling\01_Input_From_Partition\`
  - `final_audio_features/*.npy`
  - `final_visual_features/*.npy`
  - `final_text_features/*.npy`

### B. Indeks Evaluasi Terpusat Server (Global Hold-Out Test Set)
- Lokasi: `C:\Wira\Kuliah\Penelitian\clinical-xai-multimodal-depression-fl\federated_learning_output\global_test_index.csv`
- Jumlah Sampel: 56 subjek unik E-DAIC Test Set.

### C. Hasil Pengujian Kebocoran Data (Data Leakage Verification)
```python
Intersection(all_client_training_ids, global_test_ids) == set()
```
- Total Subjek Training Klien 1, 2, 3: **163 subjek unik**
- Total Subjek Global Hold-Out Test Set: **56 subjek unik**
- Irisan (Overlap): **0 subjek (Zero Leakage Terkonfirmasi 100%)**
