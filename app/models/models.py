import uuid
import enum
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Enum, ForeignKey, Float, Text, Integer
from sqlalchemy.orm import relationship

from app.database import Base


def gen_uuid():
    return str(uuid.uuid4())


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Upload(Base):
    """A single uploaded source image."""
    __tablename__ = "uploads"

    id = Column(String, primary_key=True, default=gen_uuid)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    content_type = Column(String, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    jobs = relationship("Job", back_populates="upload", cascade="all, delete-orphan")


class Job(Base):
    """A processing job created for an upload (height estimation + flythrough)."""
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=gen_uuid)
    upload_id = Column(String, ForeignKey("uploads.id"), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    upload = relationship("Upload", back_populates="jobs")
    result = relationship("Result", back_populates="job", uselist=False, cascade="all, delete-orphan")


class Result(Base):
    """Output of a completed job: height map + flythrough artifact references."""
    __tablename__ = "results"

    id = Column(String, primary_key=True, default=gen_uuid)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False, unique=True)

    # Path/URL to generated height map (image or raster file)
    height_map_path = Column(String, nullable=True)
    # Path/URL to generated 3D flythrough asset (e.g. .mp4, .glb)
    flythrough_path = Column(String, nullable=True)

    # Simple summary stats — placeholder until real ML model is wired in
    min_height_m = Column(Float, nullable=True)
    max_height_m = Column(Float, nullable=True)
    mean_height_m = Column(Float, nullable=True)

    # Calibration quality — tells you whether the numbers above are
    # trustworthy, or just an arbitrary uncalibrated guess. See
    # person2_elevation/elevation_pipeline.py's generate_elevation().
    calibration_mode = Column(String, nullable=True)      # "relative" / "global" / "terrain_aware"
    error_mean = Column(Float, nullable=True)              # avg abs error vs real SRTM (None if mode="relative")
    error_max = Column(Float, nullable=True)               # worst-case abs error vs real SRTM
    correlation = Column(Float, nullable=True)             # 0-1, how well calibrated elevation tracks real SRTM
    artifact_pixels = Column(Integer, nullable=True)       # count of excluded SRTM edge-artifact pixels
    fit_report = Column(Text, nullable=True)               # JSON string: per-terrain-class fit details

    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("Job", back_populates="result")