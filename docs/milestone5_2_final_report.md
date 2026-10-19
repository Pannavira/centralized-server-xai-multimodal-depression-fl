# FED-MIND: PHASE 4 — MILESTONE 5.2 FINAL SCIENTIFIC REPORT
## Controlled Non-IID Label Skew Experiment

> **Author**: Antigravity Autonomous Agent (Google DeepMind)  
> **Project**: FED-MIND (Centralized Server & Clinical XAI Multimodal Depression FL)  
> **Run ID**: `FL-LABEL-SKEW-20261016-141108-7865D4`  
> **Control Run ID**: `FL-RUN-20261016-130403-45D587` (M5.1 FL-IID Control)  
> **Scenario**: `non_iid_label_skew` (`controlled_label_skew_v1`)  
> **Protocol Status**: `controlled_experiment_label_skew`  
> **Date / Timestamp**: 2026-10-16T14:45:00 UTC  

---

## A. EXECUTIVE SUMMARY

Milestone 5.2 (M5.2) berhasil menyelesaikan eksperimen terfederasi **Controlled Non-IID Label Skew** selama **50 ronde** penuh dengan partisipasi 3/3 klien (100%), tanpa kegagalan (`0 failures`), mempertahankan isolasi penuh terhadap model produksi (`storage/models/fl_iid/model.pt`), dan merealisasikan **perbandingan kausal murni (pure causal comparison)** terhadap kontrol FL-IID M5.1 (`FL-RUN-20261016-130403-45D587`).

### Temuan Ilmiah Utama (Primary Scientific Findings):
1. **Divergensi Klinis Ekstrem vs Stabilitas Regresi Global**:
   - Di bawah Label Skew, metrik regresi global tampak stabil bahkan mengalami sedikit penurunan error numerik ($\text{MAE}: 4.9997 \to 4.6964$, $\Delta = -0.3033$ atau $-6.07\%$; $\text{RMSE}: 6.1402 \to 5.9384$, $\Delta = -0.2018$ atau $-3.29\%$).
   - Namun, **sensitivitas klinis terdegradasi secara kritis**: **Recall (Sensitivity)** anjlok dari **0.6667** (IID) menjadi **0.4762** (Label Skew) ($\Delta = -0.1905$, penurunan relatif **$-28.57\%$**).
2. **Lonjakan Signifikan False Negative Rate (Pasien Depresi Terlewat)**:
   - **False Negative Rate (FNR)** melonjak dari **0.3333** (IID, 7 kasus terlewat) menjadi **0.5238** (Label Skew, 11 kasus terlewat), menandakan **lebih dari separuh pasien depresi gagal terdeteksi** ketika model dilatih pada data terdistribusi heterogen tanpa mitigasi Non-IID.
   - True Positives (TP) turun dari 14 menjadi 10 ($\Delta = -4$), sedangkan True Negatives (TN) naik dari 26 menjadi 30 ($\Delta = +4$). Model terdorong menjadi lebih konservatif (condong ke arah non-depresi) karena dua dari tiga klien didominasi oleh kelas kontrol negatif.
3. **Peningkatan Client Drift Parameter**:
   - Client update drift parameter ($L_2$ norm terhadap model global masuk) teramati memiliki rata-rata keseluruhan $15.4150$, dengan puncaknya mencapai $61.3575$ pada fase awal konvergensi (Ronde 1) akibat gradien lokal yang sangat bertolak belakang antara Client 1 (79.6% positif) dan Client 2 (96.4% negatif).
4. **Akuntansi Komunikasi Identik**:
   - Beban payload komunikasi terfederasi tepat identik dengan M5.1: $9,398,401,200$ parameter bytes ($\approx 8.7529\text{ GB}$, delta $0$ bytes), membuktikan integritas transfer bobot parameter tanpa kebocoran overhead tambahan.

---

## B. M5.1 CONTROL BASELINE

Eksperimen kontrol FL-IID M5.1 (`FL-RUN-20261016-130403-45D587`) berperan sebagai tolok ukur acuan tunggal (*single primary control*):

| Variabel Eksperimen | Nilai Kontrol M5.1 |
|---|---|
| **Run ID** | `FL-RUN-20261016-130403-45D587` |
| **Inisialisasi Round 0 SHA-256** | `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973` |
| **Jumlah Klien & Partisipasi** | 3 Klien (100% full participation, 3/3 per ronde) |
| **Jumlah Sampel per Klien** | Client 1: 54, Client 2: 55, Client 3: 54 (Total = 163) |
| **Total Ronde / Local Epochs** | 50 Ronde FedAvg / 2 Local Epochs per ronde |
| **Batch Size / Optimizer / LR** | Batch size = 16, AdamW, $\text{LR} = 1\times 10^{-4}$, Weight Decay = 0.01 |
| **Fungsi Loss** | `ClassWeightedMSELoss` ($\text{pos\_weight} = 2.5, \text{fn\_penalty} = 2.0, \text{threshold} = 10.0$) |
| **Validation Cohort** | Shared validation cohort ($N=56$, SHA-256: `54a1dd5c...`) |
| **Global Hold-Out Test** | Global test index ($N=56$, SHA-256: `27ea4f24...`) |
| **M5.1 R50 Baseline Metrics** | MAE: 4.9997, RMSE: 6.1402, Macro F1: 0.7005, Recall: 0.6667, Spec: 0.7429, AUC: 0.7592 |

---

## C. HISTORICAL LABEL-SKEW AUDIT

Sebelum membuat partisi baru, audit forensik dilakukan terhadap direktori riset historis `clinical-xai-multimodal-depression-fl/federated_learning_output/NonIID_LabelSkew/`. Ditemukan bahwa partisi historis **tidak valid secara metodologis untuk eksperimen kontrol murni** karena:
1. **Severe Data Leakage**: Sebanyak 30 subjek dari `global_test_index.csv` bocor ke dalam data pelatihan klien historis.
2. **Perubahan Pool Subjek**: Total subjek pelatihan hanya 149 (65 subjek hilang, 51 subjek asing ditambahkan dari luar pool M5.1).
3. **Quantity Skew Confounder**: Jumlah sampel antar-klien tidak seragam ($N_1 = 22, N_2 = 27, N_3 = 100$). Ini menyatukan variabel Quantity Skew ke dalam Label Skew, sehingga merusak isolasi kausal.

*Kesimpulan Audit*: Protokol partisi baru yang bersih (`controlled_label_skew_v1`) diimplementasikan untuk menjamin validitas ilmiah terkontrol.

---

## D. LABEL-SKEW PARTITION DESIGN (`controlled_label_skew_v1`)

Partisi baru dibangkitkan secara deterministik menggunakan `seed=42` dari 163 subjek unik training pool M5.1 ($S_{\text{IID}} \equiv S_{\text{LabelSkew}}$):
- **Prioritas Optimasi**:
  1. Pool partisipan 100% identik dengan M5.1 ($N=163$, 50 positif, 113 negatif).
  2. Mempertahankan jumlah data per klien tepat **54 / 55 / 54** (Zero Quantity Skew).
  3. Nol duplikasi subjek antar-klien.
  4. Client 1 difokuskan sebagai **Depression-Heavy** (target proporsi positif $\approx 80\%$).
  5. Client 2 difokuskan sebagai **Control-Heavy** (target proporsi positif serendah mungkin, tetap mempertahankan kedua kelas jika memungkinkan).
  6. Client 3 menerima sisa partisipan (**Mixed/Remainder**).

---

## E. LABEL DISTRIBUTION TABLE

Distribusi label empiris yang berhasil dicapai secara deterministik pada M5.2 dibandingkan dengan M5.1 IID:

| Skenario | Klien | Total Samples | Positive ($PHQ \ge 10$) | Negative ($PHQ < 10$) | Positive % | Mean PHQ-8 | Median PHQ-8 | Std PHQ-8 | Binary Entropy |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M5.1 Control** | **Client 1** | 54 | 16 | 38 | 29.63% | 7.0556 | 6.0 | 6.5370 | 0.8767 |
| **M5.1 Control** | **Client 2** | 55 | 17 | 38 | 30.91% | 6.3818 | 5.0 | 5.3107 | 0.8921 |
| **M5.1 Control** | **Client 3** | 54 | 17 | 37 | 31.48% | 6.5370 | 3.5 | 6.5176 | 0.8987 |
| **M5.2 Label Skew** | **Client 1** | 54 | 43 | 11 | **79.63%** | 12.2222 | 12.0 | 5.8942 | 0.7293 |
| **M5.2 Label Skew** | **Client 2** | 55 | 2 | 53 | **3.64%** | 3.2727 | 2.0 | 3.4931 | 0.2254 |
| **M5.2 Label Skew** | **Client 3** | 54 | 5 | 49 | **9.26%** | 4.5370 | 3.5 | 4.3598 | 0.4451 |

### Metrik Kuantifikasi Heterogenitas (Prevalence Metrics):
- **Max - Min Positive Prevalence Difference**:
  - M5.1 IID: $31.48\% - 29.63\% = \mathbf{1.85\%}$
  - M5.2 Label Skew: $79.63\% - 3.64\% = \mathbf{75.99\%}$
- **Mean Pairwise Jensen-Shannon Divergence**:
  - M5.1 IID: $\mathbf{0.0114}$
  - M5.2 Label Skew: $\mathbf{0.4810}$ (peningkatan divergensi $> 42\times$)

---

## F. CONTROLLED EXPERIMENT INVARIANTS

| Invarian Kontrol | Persyaratan Protokol | Hasil Evaluasi M5.2 | Status |
|---|---|---|:---:|
| **Training Participant Pool** | $S_{\text{M5.2}} \equiv S_{\text{M5.1}}$ | 163 ID tepat identik, selisih himpunan kosong | **PASSED** |
| **Client Sample Quantities** | 54 / 55 / 54 | 54 / 55 / 54 dipertahankan presisi | **PASSED** |
| **Zero Duplicate ID** | Himpunan klien mutually disjoint | $\text{C1} \cap \text{C2} \cap \text{C3} = \emptyset$ | **PASSED** |
| **Global Label Counts** | Positif: 50, Negatif: 113 | Positif: 50, Negatif: 113 | **PASSED** |
| **Round 0 Checkpoint SHA-256** | `a1c052476e613d9d6795e824...` | `a1c052476e613d9d6795e824...` | **PASSED** |
| **Validation Cohort SHA-256** | `54a1dd5c7ea1a544c9b9ca...` | `54a1dd5c7ea1a544c9b9ca...` | **PASSED** |
| **Global Test Set SHA-256** | `27ea4f24fb8aa32fb6d691...` | `27ea4f24fb8aa32fb6d691...` | **PASSED** |
| **Architecture & Params** | Multimodal Baseline (7,832,001) | 7,832,001 parameters (31,328,004 B) | **PASSED** |
| **Hyperparameters** | AdamW, LR 1e-4, wd 0.01, bs 16, ep 2 | Identik dengan M5.1 | **PASSED** |

---

## G. DATA INTEGRITY & LEAKAGE ASSERTIONS

- **Train vs Test Leakage**:
  $$\text{Intersection}(S_{\text{Train\_Label\_Skew}}, S_{\text{Global\_Test}}) = \emptyset \quad (\text{Overlap} = 0)$$
- **Train vs Shared Validation Leakage**:
  $$\text{Intersection}(S_{\text{Train\_Label\_Skew}}, S_{\text{Shared\_Val}}) = \emptyset \quad (\text{Overlap} = 0)$$
- **Model Isolasi Produksi**:
  - `storage/models/fl_iid/model.pt` SHA-256 tetap `17d59aa90455386a1e2f00bed2107ef0cbe2a6fb3e96e3f476faf0272db0ec73`. Tidak mengalami modifikasi atau *overwrite*.

---

## H. GATE A RESULT (Partition Validation)
- Verifikasi otomatis seluruh 163 ID training pool terhadap partisi M5.1.
- Evaluasi nol irisan terhadap dataset uji global dan validasi.
- Hash konsisten dengan `partition_manifest.json`.
- **Hasil Gate A**: **PASSED (100% Valid)**.

---

## I. GATE B RESULT (3-Round Integration Smoke Test)
- Pengujian eksekusi simulasi 3 ronde FL pada partisi `non_iid_label_skew`.
- Validasi transmisi aman `MetricRecord` dan pembaharuan tensor `ArrayRecord`.
- Verifikasi pencatatan metrik pada `client_metrics.csv` dan ketiadaan kebocoran ID partisipan melalui jaringan.
- **Hasil Gate B**: **PASSED (100% Valid)**.

---

## J. 50-ROUND EXECUTION
- **Waktu Eksekusi Penuh**: ~22 menit.
- **Ronde Selesai**: 50 / 50 ronde.
- **Partisipasi Klien**: 3/3 klien per ronde (100% kehadiran pada seluruh 50 ronde).
- **Kegagalan / Dropouts**: 0 kegagalan numerik, loss selalu berhingga (*finite*).
- **Semua Checkpoint**: 51 checkpoint tersimpan utuh (`round_000` hingga `round_050`).

---

## K. CONVERGENCE ANALYSIS
- **Loss Konvergensi Global**: Loss uji hold-out global berawal dari $206.87$ pada Ronde 0, turun secara konsisten di bawah $50.0$ pada Ronde 10, dan berfluktuasi stabil di kisaran $30.0 - 45.0$ pada paruh akhir pelatihan.
- Fluktuasi kurva loss pada Label Skew sedikit lebih bervariasi dibanding IID akibat tarik-menarik gradien dari klien berpopulasi mayoritas berlawanan.

---

## L. CLIENT DRIFT ANALYSIS

Analisis deviasi parameter lokal terhadap model global ($||W_{\text{local}} - W_{\text{global}}||_2$):
- **Overall Mean Drift**: **15.4150**
- **Overall Max Drift**: **61.3575** (terjadi pada Ronde 1)
- **Round 01 Mean Drift**: **60.7880**
- **Round 50 Mean Drift**: **14.0624**
- **Interpretasi**:
  Pada ronde-ronde awal, ketika bobot model global masih dekat dengan inisialisasi acak, gradien Client 1 (yang sarat pasien depresi berat) mengarahkan bobot secara agresif ke estimasi skor PHQ-8 tinggi, sedangkan Client 2 (kontrol sehat) mengarahkan bobot ke skor rendah. Akibatnya terjadi lonjakan drift parameter sebesar $>60$. Seiring berjalannya ronde dan diterapkannya averaging FedAvg, model global menemukan kompromi parameter dan drift menyusut serta stabil di kisaran $13.5 - 15.0$ pada ronde 40-50.

Grafik trajektori client drift per klien telah dihasilkan pada `plots/client_drift_label_skew.png`.

---

## M. FALSE NEGATIVE ANALYSIS (CLINICAL SAFETY CRITICAL)

Analisis kesalahan klasifikasi klinis (Threshold $PHQ-8 \ge 10.0$):

| Metrik Kesalahan Klinis | M5.1 FL-IID Control | M5.2 Non-IID Label Skew | Selisih Mutlak ($\Delta$) | Interpretasi Klinis |
|---|:---:|:---:|:---:|---|
| **True Positives (TP)** | 14 | 10 | **-4** | 4 pasien depresi kehilangan deteksi positif |
| **False Negatives (FN)** | 7 | 11 | **+4** | **Kasus depresi terlewat meningkat 57%** |
| **True Negatives (TN)** | 26 | 30 | **+4** | Deteksi individu sehat meningkat |
| **False Positives (FP)** | 9 | 5 | **-4** | False alarm berkurang |
| **False Negative Rate (FNR)** | **33.33%** ($0.3333$) | **52.38%** ($0.5238$) | **+19.05%** | **FNR melonjak signifikan: >50% depresi terlewat** |
| **False Positive Rate (FPR)** | **25.71%** ($0.2571$) | **14.29%** ($0.1429$) | **-11.43%** | Spesifisitas meningkat mengorbankan sensitivitas |

> [!CAUTION]
> **Temuan Klinis Sangat Kritis**:
> Meskipun error regresi rata-rata (MAE/RMSE) tampak membaik secara artifisial, model yang dilatih pada Label Skew tanpa mitigasi mengalami **Clinical Blindness** terhadap pasien depresi. Model menjadi terlalu ragu (*under-predicting*) karena dua dari tiga klien mengajarkan bahwa depresi adalah kondisi langka ($3.6\%$ dan $9.3\%$). Akibatnya, pada ambang klinis $PHQ \ge 10.0$, mayoritas pasien depresi aktual tergolong sebagai non-depresi (False Negative Rate naik menjadi $52.38\%$). Ini mengonfirmasi hipotesis protokol penelitian bahwa Recall/Sensitivity adalah metrik evaluasi primer terpenting dalam skrining depresi.

---

## N. IID VS LABEL-SKEW COMPARISON (ROUND 50)

Hasil komparasi formal model akhir Ronde 50 pada dataset uji hold-out global ($N=56$):

| Metrik Evaluasi | M5.1 IID Control | M5.2 Label Skew | Selisih Mutlak ($\Delta$) | Selisih Relatif (%) | Karakter Metrik |
|---|:---:|:---:|:---:|:---:|:---:|
| **MAE** | 4.9997 | 4.6964 | -0.3033 | -6.07% | Lower is better |
| **RMSE** | 6.1402 | 5.9384 | -0.2018 | -3.29% | Lower is better |
| **MSE** | 37.7017 | 35.2650 | -2.4367 | -6.46% | Lower is better |
| **Macro F1-Score** | 0.7005 | 0.6725 | -0.0280 | -4.00% | Higher is better |
| **Recall / Sensitivity** | **0.6667** | **0.4762** | **-0.1905** | **-28.57%** | **Higher is better** |
| **Specificity** | 0.7429 | 0.8571 | +0.1142 | +15.37% | Higher is better |
| **ROC-AUC** | 0.7592 | 0.7116 | -0.0476 | -6.27% | Higher is better |
| **Lin's CCC** | 0.4345 | 0.3894 | -0.0451 | -10.38% | Higher is better |
| **Pearson Correlation ($r$)** | 0.4524 | 0.4286 | -0.0238 | -5.26% | Higher is better |
| **False Negative Rate (FNR)** | **0.3333** | **0.5238** | **+0.1905** | **+57.16%** | **Lower is better** |
| **False Positive Rate (FPR)** | 0.2571 | 0.1429 | -0.1143 | -44.46% | Lower is better |

*Catatan*: Seluruh selisih dinyatakan sebagai *observed difference* pada seed 42, bukan klaim signifikansi statistik lintas seed ganda.

---

## O. LATE-ROUND STABILITY (ROUNDS 46–50)

Evaluasi konsistensi model pada 5 ronde terakhir:

| Metrik | Trajektori M5.1 IID (Mean $\pm$ Std) | Trajektori M5.2 Label Skew (Mean $\pm$ Std) |
|---|:---:|:---:|
| **MAE** | $5.2407 \pm 0.4543$ | $4.8860 \pm 0.3226$ |
| **Macro F1** | $0.6635 \pm 0.0641$ | $0.6560 \pm 0.0364$ |
| **Recall** | $0.7048 \pm 0.1137$ | $0.5714 \pm 0.1214$ |

Label Skew menunjukkan stabilitas nilai standar deviasi yang sedikit lebih ketat pada MAE dan Macro F1 di akhir ronde, namun mempertahankan penurunan tingkat Recall yang konsisten ($\sim 0.57$ vs $\sim 0.70$).

---

## P. COMMUNICATION COMPARISON

| Aspek Komunikasi | M5.1 IID Control | M5.2 Label Skew | Selisih ($\Delta$) |
|---|:---:|:---:|:---:|
| **Total Parameter per Model** | 7,832,001 | 7,832,001 | 0 |
| **Ukuran Model (FP32)** | 31,328,004 bytes (~29.8767 MB) | 31,328,004 bytes (~29.8767 MB) | 0 bytes |
| **Payload per Ronde (3 Klien Down/Up)** | 187,968,024 bytes (~179.2603 MB) | 187,968,024 bytes (~179.2603 MB) | 0 bytes |
| **Total Payload 50 Ronde** | 9,398,401,200 bytes (~8.7529 GB) | 9,398,401,200 bytes (~8.7529 GB) | **0 bytes** |

Grafik perbandingan beban transmisi kumulatif telah disimpan pada `plots/communication_iid_vs_label_skew.png`.

---

## Q. CHECKPOINT & REPRODUCIBILITY INTEGRITY

1. Seluruh 51 checkpoint dari ronde 000 hingga 050 tersimpan lengkap:
   `storage/federated/runs/FL-LABEL-SKEW-20261016-141108-7865D4/rounds/round_000/model.pt` s.d. `round_050/model.pt`.
2. Checkpoint `round_000/model.pt` memiliki checksum SHA-256 yang tepat sama dengan M5.1:
   `a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973`.
3. Checkpoint final `round_050/model.pt` berhasil dimuat dengan parameter `strict=True` ke dalam `MultimodalBaselineModel` tanpa *unexpected keys* atau *missing keys*.
4. Tidak ada berkas M5.1 atau model produksi `storage/models/fl_iid/model.pt` yang tertimpa.

---

## R. PRIVACY VALIDATION

Kontrak privasi Flower 1.40 Message API terverifikasi 100% dipatuhi selama 50 ronde:
- **Terkirim via Jaringan**: HANYA `ArrayRecord` (bobot model) dan `MetricRecord` (num-examples, train_loss, train_mae, train_f1, val_loss, val_mae, val_f1, client_drift_l2).
- **TIDAK PERNAH Terkirim**: ID partisipan, daftar label PHQ-8, jumlah positif/negatif, audio/video/transkrip mentah, tensor fitur multimodal, prediksi per pasien, atau trace gradien individual.

---

## S. AUTOMATED TEST SUITE EXECUTION

Pengujian unit dan integrasi otomatis dieksekusi dengan `pytest -v tests`:
```text
======================= 40 passed, 1 warning in 51.59s ========================
```
- 28 pengujian eksisting (FedAvg, Flower Client/Server/Checkpointing, API, Skrining, Golden Regression P600) tetap **100% lulus**.
- 12 pengujian baru khusus Milestone 5.2 (`tests/test_federated_label_skew.py`) **100% lulus**:
  1. `test_1_iid_and_label_skew_pool_equality_and_no_duplication` PASSED
  2. `test_2_exact_client_quantities_and_global_class_counts` PASSED
  3. `test_3_label_skew_prevalence_difference_exceeds_iid` PASSED
  4. `test_4_zero_leakage_and_hash_identity_with_m51_control` PASSED
  5. `test_5_round_zero_initialization_sha256_match` PASSED
  6. `test_6_deterministic_partition_generation_under_seed_42` PASSED
  7. `test_7_label_distribution_manifest_correctness` PASSED
  8. `test_8_fnr_fpr_clinical_metric_calculation` PASSED
  9. `test_9_client_drift_l2_norm_finite` PASSED
  10. `test_10_production_model_isolation_preserved` PASSED
  11. `test_11_expected_communication_payload_identical_to_m51` PASSED
  12. `test_12_label_skew_three_round_integration_smoke` PASSED

---

## T. SCIENTIFIC LIMITATIONS

1. **Single Random Seed (Seed 42)**: Eksperimen ini mengevaluasi satu trajektori pelatihan deterministik pada seed 42. Hasil dilaporkan sebagai perbedaan observasi kausal terkontrol (*observed difference*), bukan kesimpulan signifikansi statistik general yang memerlukan multi-seed repetitions.
2. **Karakter Non-IID Terisolasi**: Sesuai mandat metodologis M5.2, eksperimen ini secara eksklusif memvariasikan prevalensi label (*Label Skew* murni) dan sengaja menahan variasi kuantitas (*Quantity Skew*) serta modalitas (*Modality Skew*).
3. **Ketiadaan Algoritma Mitigasi**: Eksperimen ini berfungsi sebagai *benchmark diagnosis* dampak heterogenitas label dasar. Algoritma mitigasi Non-IID seperti FedProx, SCAFFOLD, atau loss re-weighting dinamis belum diuji pada milestone ini (dialokasikan pada fase selanjutnya).

---

## U. FINAL STATUS

```text
============================================================
M5.2 EXECUTION PASSED
============================================================
```

Semua kriteria penerimaan eksperimen ilmiah terkontrol M5.2 telah terpenuhi secara sempurna tanpa deviasi protokol.
