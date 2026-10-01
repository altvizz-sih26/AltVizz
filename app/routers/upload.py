import json
import os
import shutil
import uuid

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.models.models import Upload, Job, JobStatus, Result
from app.schemas import JobOut
from app.services.ml_stub import run_height_estimation
from app.services.depth_pipeline import run_compare_pipeline

router = APIRouter(prefix="/api/upload", tags=["upload"])

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/tiff"}


def _update_job_progress(db: Session, job_id: str, stage: str, pct: int):
    job = db.query(Job).filter(Job.id == job_id).first()
    if job is None:
        return
    job.stage = stage
    job.progress = max(0, min(100, int(pct)))
    db.commit()


def _store_upload(file: UploadFile):
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1]
    filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
    with open(filepath, "wb") as destination:
        shutil.copyfileobj(file.file, destination)
    return filename, filepath


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
        job.stage = "Starting"
        job.progress = 0
        db.commit()

        try:
            output = run_height_estimation(
                filepath,
                srtm_path=srtm_path,
                progress=lambda stage, pct: _update_job_progress(db, job_id, stage, pct),
            )
            result = Result(job_id=job_id, **output)
            db.add(result)
            job.status = JobStatus.COMPLETED
            job.stage = "Completed"
            job.progress = 100
            db.commit()
        except Exception as e:
            job.status = JobStatus.FAILED
            job.stage = "Failed"
            job.error_message = str(e)
            db.commit()
    finally:
        db.close()


def process_compare_job(
    job_id: str,
    before_path: str,
    after_path: str,
    srtm_path: str | None = None,
):
    """Background task for a before/after comparison job."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return
        job.status = JobStatus.PROCESSING
        job.stage = "Starting"
        job.progress = 0
        db.commit()

        try:
            output = run_compare_pipeline(
                before_path,
                after_path,
                srtm_path=srtm_path,
                progress=lambda stage, pct: _update_job_progress(db, job_id, stage, pct),
            )
            result = Result(
                job_id=job_id,
                flythrough_path=output["after_flythrough_path"],
                before_dsm_geotiff_path=output["before_dsm_geotiff_path"],
                after_dsm_geotiff_path=output["after_dsm_geotiff_path"],
                analysis_path=output["analysis_path"],
                before_flythrough_path=output["before_flythrough_path"],
                change_summary=json.dumps(output["change_summary"]),
                calibration_mode=output["calibration_mode"],
            )
            db.add(result)
            job.status = JobStatus.COMPLETED
            job.stage = "Completed"
            job.progress = 100
            db.commit()
        except Exception as e:
            job.status = JobStatus.FAILED
            job.stage = "Failed"
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


@router.post("/compare", response_model=JobOut)
async def upload_compare(
    background_tasks: BackgroundTasks,
    before_file: UploadFile = File(...),
    after_file: UploadFile = File(...),
    srtm_file: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    """Upload a before/after pair and start an asynchronous comparison."""
    files = [before_file, after_file]
    if srtm_file is not None:
        files.append(srtm_file)
    for file in files:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Allowed: {ALLOWED_CONTENT_TYPES}",
            )

    before_name, before_path = _store_upload(before_file)
    after_name, after_path = _store_upload(after_file)
    srtm_path = _store_upload(srtm_file)[1] if srtm_file is not None else None

    before_upload = Upload(
        filename=before_name,
        filepath=before_path,
        content_type=before_file.content_type,
    )
    after_upload = Upload(
        filename=after_name,
        filepath=after_path,
        content_type=after_file.content_type,
    )
    db.add_all([before_upload, after_upload])
    db.commit()
    db.refresh(before_upload)
    db.refresh(after_upload)

    job = Job(
        upload_id=before_upload.id,
        after_upload_id=after_upload.id,
        kind="compare",
        status=JobStatus.PENDING,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(
        process_compare_job,
        job.id,
        before_path,
        after_path,
        srtm_path,
    )
    return job