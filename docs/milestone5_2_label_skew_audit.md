# Milestone 5.2: Audit Historis Partisi & Eksperimen Non-IID Label Skew

**Dokumen**: `docs/milestone5_2_label_skew_audit.md`  
**Proyek**: FED-MIND (Centralized Server & Federated Learning Infrastructure)  
**Waktu Audit**: 2026-10-16  
**Repositori Referensi**: `C:\Wira\Kuliah\Penelitian\clinical-xai-multimodal-depression-fl`  
**Status Audit**: **AUDIT SELESAI — PARTISI HISTORIS DINYATAKAN CACAT (DEFEKTIF)**  

---

## 1. Ringkasan Eksekutif Audit

Audit mendalam dilakukan terhadap direktori partisi Non-IID historis `federated_learning_output/NonIID_LabelSkew/`, log eksperimen `experiments/fedavg_noniid_labelskew_randominit/`, `experiments_v2/fedavg_noniid_labelskew_pretrained/`, dan berkas laporan Excel `label_skew_distribution_report.xlsx`.

### Temuan Kritis:
Partisi historis `NonIID_LabelSkew` **TIDAK DAPAT DIGUNAKAN** untuk eksperimen kontrol Milestone 5.2 karena mengandung cacat metodologis fatal:
1. **Kebocoran Data Kritis (Test Set Leakage)**: Sebanyak **30 subjek** dari `global_test_index.csv` (hold-out global test) bocor ke dalam data training klien (`client_1_train.csv`, `client_2_train.csv`, `client_3_train.csv`).
2. **Perubahan Kuantitas Klien Ekstrem (Confounded with Quantity Skew)**: Jumlah sampel per klien berubah drastis menjadi `22`, `27`, dan `100` (bukan `54`, `55`, `54`). Hal ini mencampuradukkan pengaruh skew label dengan dominasi ukuran klien (Quantity Skew).
3. **Ketidaksamaan Kohort Training ($S_{\text{IID}} \neq S_{\text{LabelSkew}}$)**: Dari 163 partisipan resmi M5.1, sebanyak 65 partisipan hilang dan 51 partisipan non-training ditambahkan, menyisakan total 149 partisipan.
4. **Cacat Pemilihan Model Historis**: Script historis menghentikan run pada ronde 6 (randominit) dan memilih best model menggunakan data test global.

### Keputusan Ilmiah:
Sesuai protokol penelitian dan panduan instruksi eksperimen, partisi historis ditolak. Milestone 5.2 **WAJIB** menggunakan protokol partisi terkontrol baru:
$$\mathbf{partition\_protocol} = \mathbf{"controlled\_label\_skew\_v1"}$$
yang secara ketat menjaga populasi 163 partisipan M5.1, kuantitas 54/55/54, dan zero-leakage 100%.

---

## 2. Matriks Komparasi Audit Partisi Historis

| Parameter Evaluasi | Nilai pada M5.1 (IID Control) | Nilai Historis `NonIID_LabelSkew` | Status Kepatuhan Kontrol |
|---|:---:|:---:|:---:|
| **Total Subjek Training** | 163 partisipan | 149 partisipan (65 hilang, 51 asing) | ❌ **FAIL (Pool Incompatible)** |
| **Kuantitas Klien 1** | 54 partisipan | 22 partisipan | ❌ **FAIL (Quantity Skew Confounder)** |
| **Kuantitas Klien 2** | 55 partisipan | 27 partisipan | ❌ **FAIL (Quantity Skew Confounder)** |
| **Kuantitas Klien 3** | 54 partisipan | 100 partisipan | ❌ **FAIL (Quantity Skew Confounder)** |
| **Kontaminasi Data Uji Global** | 0 subjek (Zero Leakage) | **30 subjek bocor** (`P600`, `P602`, `P604`, `P605`, `P716`, dll.) | ❌ **CRITICAL DEFECT (Leakage)** |
| **Algoritma Partisi** | Terstratifikasi IID | Skew heuristik tidak terkontrol | ❌ **FAIL** |
| **Jumlah Ronde Historis** | 50 Ronde penuh (M5.1) | 6 Ronde (v1) / 20 Ronde (v2) | ❌ **Incomplete Trajectory** |
| **Inisialisasi Model Historis** | Seed 42 RandomInit (`a1c052...`) | RandomInit (v1) / Pretrained `modeling2` (v2) | ⚠️ **Divergent Initialization** |

---

## 3. Rincian Kebocoran Subjek Test Set pada Partisi Historis

Berikut adalah 30 ID partisipan `global_test_index.csv` yang terkontaminasi masuk ke dalam training set `NonIID_LabelSkew` historis:
```text
['600', '602', '604', '605', '609', '619', '620', '629', '636', '637',
 '638', '640', '651', '656', '658', '663', '664', '666', '669', '676',
 '682', '683', '689', '693', '699', '705', '710', '712', '716', '717']
```
Subjek-subjek ini termasuk kasus acuan kritis (seperti `P600` Borderline, `P605` Non-risk, dan `P716` High Risk). Keberadaan data ini di dalam data latih klien menyebabkan evaluasi global test pada eksperimen historis kehilangan validitas ilmiahnya (*evaluation integrity breach*).

---

## 4. Spesifikasi Desain Partisi Terkontrol `controlled_label_skew_v1`

Untuk memastikan eksperimen Milestone 5.2 merupakan uji kausal yang valid (*pure single-variable controlled experiment*):
1. **Target Desain**:
   - **Client 1**: *Depression-Heavy* (prevalensi positif dimaksimalkan).
   - **Client 2**: *Control-Heavy* (prevalensi kontrol/negatif dimaksimalkan).
   - **Client 3**: *Mixed / Remainder* (menampung sisa subjek secara deterministic).
2. **Invarian Wajib**:
   - $S_{\text{M5.2}} \equiv S_{\text{M5.1}}$ (163 subjek identik 100%).
   - Kuantitas klien dipertahankan persis: Client 1 = 54, Client 2 = 55, Client 3 = 54.
   - Zero-leakage: $S_{\text{train}} \cap S_{\text{global\_test}} = \emptyset$ dan $S_{\text{train}} \cap S_{\text{val}} = \emptyset$.
   - Tidak ada oversampling, undersampling, atau data sintetis.
   - Initial round-0 checkpoint: `storage/federated/runs/FL-RUN-20261016-130403-45D587/rounds/round_000/model.pt` (SHA-256: `a1c052476e...`).
