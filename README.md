# FED-MIND Centralized Server (Phase 1 MVP)

Backend Centralized Application Server untuk deteksi depresi klinis multimodal berbasis deep learning dan Explainable AI (XAI).

## Arsitektur

```text
               APPLICATION PLANE
Flutter / Client ───────► FastAPI ───────► Global Model
                             │
                             └────────────► XAI

               FEDERATED PLANE (Phase 3)
FL Client 1 ─┐
FL Client 2 ─┼──► Flower Server ───► Global Model
FL Client 3 ─┘
```

## Menjalankan Server

### Opsi 1: Menggunakan Skrip Praktis (Windows)

```bash
run_server.bat
```

### Opsi 2: Menggunakan Uvicorn Langsung

Jika menggunakan venv dari repo riset:

```powershell
& "..\..\clinical-xai-multimodal-depression-fl\.venv\Scripts\uvicorn.exe" app.main:app --reload --host 127.0.0.1 --port 8000
```

Atau jika venv sudah diaktifkan di terminal:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Endpoints Utama

### Milestone 1: Server & Model Diagnostics

1. `GET /health` : Monitoring kesiapan server dan status model di CUDA.
2. `GET /model/info` : Metadata arsitektur model multimodal baseline.
3. `http://127.0.0.1:8000/docs` : Dokumentasi interaktif Swagger UI.

### Milestone 2 & 2.5: Inferensi Skrining, Exact XAI, & Scientific Consistency

4. `POST /screenings/predict`
   Endpoint langsung untuk triase skrining depresi multimodal berbasis input pre-extracted feature tensors (backward-compatible).

   **Karakteristik & Audit Ilmiah:**
   - **Stratifikasi Risiko Presisi:**
     - `< 8.0`: Low (Non-Risk)
     - `8.0 – 9.9`: Borderline (Zona Pengawasan Klinis)
     - `10.0 – 14.9`: Moderate (Terindikasi Depresi)
     - `>= 15.0`: High (Depresi Berat)
     - Keputusan triase biner: `depression_risk` jika skor $\ge 10.0$, `non_depression` jika $< 10.0$.
   - **Risk Probability Proxy:** Menghitung proxy probabilitas berbasis sigmoid fungsi skor terpusat di ambang 10.0 (tidak mengklaim kalibrasi formal sebelum kalibrasi empiris dievaluasi).
   - **Exact Modality Shapley Value (8 Koalisi):** Komputasi atribusi modalitas eksak per-pasien ($2^3 = 8$ koalisi laten) dengan baseline $v_{000}$ dan baseline latents yang identik dengan model aktif.
   - **Isolasi Model & Baseline:** Checkpoint dan latent baseline disimpan terpisah per model:
     - `storage/models/centralized/` (`model.pt`, `xai_baseline.pt`)
     - `storage/models/fl_iid/` (`model.pt`, `xai_baseline.pt`)
     - `storage/models/fl_non_iid/` (`model.pt`, `xai_baseline.pt`)
   - **Audit Feature-Level XAI:** Menetapkan `feature_xai: null` sementara sampai pipeline continuous KernelSHAP diimplementasikan secara empiris.
   - **Privacy-First Data Retention:** Tensor fitur multimodal (`512x80`, `512x72x3`, `512x768`) hanya diproses di memori dan langsung dibuang, tidak disimpan ke database.

### Milestone 3: Screening Lifecycle & Persistence (SQLAlchemy 2)

5. `POST /screenings`
   Membuat sesi skrining baru (`status: "created"`).
6. `POST /screenings/{screening_id}/predict`
   Menjalankan inferensi multimodal + Exact XAI untuk sesi yang sudah dibuat, mencatat hasil ke database, dan mengupdate status menjadi `"completed"`.
7. `GET /screenings/{screening_id}`
   Mengambil detail hasil skrining, estimasi skor, atribusi modalitas eksak, narasi klinis, dan metadata model.
8. `GET /screenings`
   Mengambil riwayat skrining dengan pagination (`skip`, `limit`) dan filter status untuk kebutuhan layar riwayat aplikasi klien.

## Database & Struktur Entitas

Sistem menggunakan SQLAlchemy 2 dengan database default SQLite (`storage/fedmind.db`) dan dukungan penuh PostgreSQL:

1. `screenings`: Sesi skrining, ID subjek, status (`created`, `completed`, `failed`), timestamp.
2. `screening_results`: Estimasi PHQ-8, kategori risiko, triase biner, proxy probabilitas.
3. `xai_results`: Nilai Shapley eksak per modalitas, modalitas dominan, baseline value, dan full JSON report.
4. `model_versions`: Registry checkpoint, nama model, checksum SHA-256, baseline path.
5. `audit_logs`: Log jejak audit setiap event eksekusi inferensi.

## Pengujian & Utilitas

### Pengujian Otomatis (Pytest)

Menjalankan seluruh 10 test case otomatis (termasuk verifikasi formula, Quality Flags, Golden Regression Test Kasus P600, dan Lifecycle Flow):

```powershell
& "..\..\clinical-xai-multimodal-depression-fl\.venv\Scripts\pytest.exe" -v tests
```

### CLI Pengujian Multi-Pasien

Menguji berbagai profil subjek (P600 Borderline, P605 Low, P716 High, dll):

```powershell
& "..\..\clinical-xai-multimodal-depression-fl\.venv\Scripts\python.exe" scripts/test_patient.py --pid 600
```
