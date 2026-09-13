import os
import shutil
import uuid

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.models.models import Upload, Job, JobStatus, Result
from app.schemas import JobOut
from app.services.ml_stub import run_height_estimation

router = APIRouter(prefix="/api/upload", tags=["upload"])

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/tiff"}


def process_job(job_id: str, filepath: str, srtm_path: str | None = None):
    """
    Background task: run height estimation and store the result.

    Opens its OWN DB session rather than reusing the request's session,
    since the request session is closed once the response is returned
    and background tasks may run after that.
    """
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return
        job.status = JobStatus.PROCESSING
        db.commit()

        try:
            output = run_height_estimation(filepath, srtm_path=srtm_path)
            result = Result(job_id=job_id, **output)
            db.add(result)
            job.status = JobStatus.COMPLETED
            db.commit()
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            db.commit()
    finally:
        db.close()


@router.post("", response_model=JobOut)
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    srtm_file: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    """
    Upload a single image and kick off a processing job.
    Returns the created job immediately with status=pending;
    poll GET /api/jobs/{job_id} for progress and results.

    `srtm_file` is optional: a matching SRTM elevation GeoTIFF for this
    image's location. If provided, calibration runs in "terrain_aware"
    mode (or "global" mode) using real ground-truth elevation. If omitted,
    calibration falls back to "relative" mode - a fixed, arbitrary 0-50m
    scale that is NOT a real height estimate. See ml_stub.py / depth_pipeline.py.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {ALLOWED_CONTENT_TYPES}",
        )

    ext = os.path.splitext(file.filename)[1]
    stored_name = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(UPLOAD_DIR, stored_name)

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    srtm_path = None
    if srtm_file is not None:
        srtm_ext = os.path.splitext(srtm_file.filename)[1]
        srtm_stored_name = f"{uuid.uuid4()}{srtm_ext}"
        srtm_path = os.path.join(UPLOAD_DIR, srtm_stored_name)
        with open(srtm_path, "wb") as f:
            shutil.copyfileobj(srtm_file.file, f)

    upload = Upload(filename=file.filename, filepath=filepath, content_type=file.content_type)
    db.add(upload)
    db.commit()
    db.refresh(upload)

    job = Job(upload_id=upload.id, status=JobStatus.PENDING)
    db.add(job)
    db.commit()
    db.refresh(job)

    # Run the (stubbed) ML pipeline in the background so the request returns instantly
    background_tasks.add_task(process_job, job.id, filepath, srtm_path)

    return job