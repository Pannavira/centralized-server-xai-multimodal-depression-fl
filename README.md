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

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Endpoints Utama (Milestone 1)

1. `GET /health`
   ```json
   {
     "status": "ok",
     "service": "FED-MIND Centralized Server",
     "model_loaded": true,
     "device": "cuda"
   }
   ```

2. `GET /model/info`
   ```json
   {
     "model": "FED-MIND Multimodal",
     "architecture": "Centralized Multimodal Baseline",
     "modalities": ["text", "audio", "visual"],
     "device": "cuda",
     "status": "ready"
   }
   ```

3. Dokumentasi Interaktif Swagger UI:
   `http://127.0.0.1:8000/docs`
