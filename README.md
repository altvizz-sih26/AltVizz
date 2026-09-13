# DepthWizard Backend (SIH26176)

FastAPI backend skeleton for single-view height estimation + 3D flythrough
generation. Built so the rest of the team (frontend, ML) can start working
against a stable API contract immediately, before the real ML model is wired in.

## What's here

- **Image upload** (`POST /api/upload`) — accepts jpeg/png/tiff, stores the file,
  creates a `Job`, and kicks off processing in the background (non-blocking).
- **Job polling** (`GET /api/jobs/{job_id}`) — check status (`pending` →
  `processing` → `completed`/`failed`) and get the result once done.
- **Job listing** (`GET /api/jobs`) — all jobs, most recent first.
- **Database models** — `Upload`, `Job`, `Result` (SQLite for now, one-line
  swap to Postgres later — see `app/database.py`).
- **Stubbed ML pipeline** (`app/services/ml_stub.py`) — returns fake height
  stats + fake file paths after a short delay, matching the exact shape the
  real model will need to return. Swap this one function out later; nothing
  else changes.

## Run it

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
uvicorn app.main:app --reload
```

Then visit `http://localhost:8000/docs` for interactive Swagger docs (auto-generated
by FastAPI — great for the frontend team to test against without needing you around).

## Try it

```bash
# Upload an image
curl -X POST http://localhost:8000/api/upload -F "file=@some_image.jpg"
# => {"id": "<job_id>", "status": "pending", ...}

# Poll for status/result
curl http://localhost:8000/api/jobs/<job_id>
```

## Next steps (in suggested order)

1. **Wire in the real ML model** in `app/services/ml_stub.py` — swap
   `run_height_estimation()`'s body for the actual monocular depth estimation
   + calibration + height-map generation. Keep the return dict shape identical.
2. **3D flythrough generation** — once height maps are real, generate a mesh
   and camera path, render to video/`.glb`. Store the output path in `Result`.
3. **Serve result files** — mount `/static` to actually serve the generated
   height maps / flythrough files (currently just placeholder paths).
4. **Auth** (if needed) — add a user model + login if multiple users need
   separate upload histories.
5. **Move to Postgres** before deployment — change `DATABASE_URL` in
   `app/database.py`, nothing else changes.
6. **Deployment** — Render/Railway for the API, since no GPU is required for
   most depth models to run inference (though training would need one).

## Project structure

```
app/
├── main.py           # FastAPI app, CORS, router registration
├── database.py        # SQLAlchemy engine/session (SQLite, swappable)
├── schemas.py          # Pydantic request/response models
├── models/
│   └── models.py       # SQLAlchemy tables: Upload, Job, Result
├── routers/
│   ├── upload.py        # POST /api/upload
│   └── jobs.py           # GET /api/jobs, /api/jobs/{id}
└── services/
    └── ml_stub.py         # Stubbed height estimation (replace with real model)
```
