import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database import Base, engine
from app.routers import upload, jobs

# Create tables on startup (fine for dev; use Alembic migrations later for prod)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="DepthWizard API",
    description="Backend for single-view height estimation + 3D flythrough generation (SIH26176)",
    version="0.1.0",
)

# Allow the frontend dev server to call this API. Tighten this before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(jobs.router)

# Serves generated height maps at the paths returned in Result.height_map_path
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
app.mount("/static/results", StaticFiles(directory=RESULTS_DIR), name="results")


@app.get("/")
def root():
    return {"status": "ok", "service": "DepthWizard API"}


@app.get("/health")
def health():
    return {"status": "healthy"}