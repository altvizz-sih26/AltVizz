from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class UploadOut(BaseModel):
    id: str
    filename: str
    uploaded_at: datetime

    class Config:
        from_attributes = True


class JobOut(BaseModel):
    id: str
    upload_id: str
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ResultOut(BaseModel):
    job_id: str
    height_map_path: Optional[str] = None
    flythrough_path: Optional[str] = None
    min_height_m: Optional[float] = None
    max_height_m: Optional[float] = None
    mean_height_m: Optional[float] = None

    # Calibration quality - tells you whether the height values above are
    # trustworthy (real SRTM-calibrated) or just an arbitrary uncalibrated
    # guess (mode="relative", no error metrics available).
    calibration_mode: Optional[str] = None
    error_mean: Optional[float] = None
    error_max: Optional[float] = None
    correlation: Optional[float] = None
    artifact_pixels: Optional[int] = None
    fit_report: Optional[str] = None  # JSON string; parse client-side if needed

    class Config:
        from_attributes = True


class JobWithResultOut(JobOut):
    result: Optional[ResultOut] = None