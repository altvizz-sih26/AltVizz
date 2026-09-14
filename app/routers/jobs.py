from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.models import Job
from app.schemas import JobWithResultOut

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobWithResultOut)
def get_job(job_id: str, db: Session = Depends(get_db)):
    """Poll this to check job status; `result` is populated once status=completed."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("", response_model=list[JobWithResultOut])
def list_jobs(db: Session = Depends(get_db)):
    """List all jobs, most recent first."""
    return db.query(Job).order_by(Job.created_at.desc()).all()
