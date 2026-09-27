"""
Height-estimation entry point.

Was a random-number stub; now delegates to the real pipeline in
depth_pipeline.py (Depth Anything V2 -> terrain classification ->
elevation calibration). Kept as a thin wrapper so the function
signature/contract this module always exposed doesn't change for
whatever else imports it.

If the depth model checkpoint isn't present yet (see depth_pipeline.py's
CHECKPOINT_PATH), this will raise FileNotFoundError with instructions -
upload.py's background task catches that and marks the job FAILED with
the error message, so you'll see it via GET /api/jobs/{id} rather than
a silent crash.
"""
from app.services.depth_pipeline import run_pipeline


def run_height_estimation(image_path: str, srtm_path: str | None = None) -> dict:
    """
    Runs the real height-estimation pipeline on an uploaded image.

    `srtm_path` is optional ground-truth elevation data for terrain-aware
    calibration (see person2_elevation/elevation_pipeline.py). Nothing
    upstream currently supplies this per-upload (would need the user to
    provide a location so the right SRTM tile can be fetched) - so for
    now every upload runs in "relative" mode: a plausible but uncalibrated
    height scale. Wiring in real SRTM lookup per upload is a good next step.

    Returns a dict matching the Result model's fields.
    """
    return run_pipeline(image_path, srtm_path=srtm_path)