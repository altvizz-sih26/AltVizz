"""
Real height-estimation pipeline: Depth Anything V2 -> terrain classification
-> elevation calibration.

Replaces the fake random output in ml_stub.py. Wires together three
previously-separate pieces of the project:

  1. depth_anything_v2/           (Person 1) - monocular depth estimation
  2. terrain_classifier/          - OpenCV terrain classification (no ML,
                                    no checkpoint needed)
  3. person2_elevation/           (Person 2) - depth -> real-world elevation
                                    calibration, optionally against SRTM

None of these three live in a proper installable package today, so this
module adds their folders to sys.path once at import time and then imports
them normally. If the project structure changes, only SIH_ROOT below needs
updating.

REQUIRES: checkpoints/depth_anything_v2_<encoder>.pth to exist at the repo
root (NOT committed to git - too large). Download separately from the
Depth Anything V2 releases and drop it in a `checkpoints/` folder before
this will run end-to-end. See README for the exact file the team is using.
"""
import json
import os
import sys
import threading

import cv2
import numpy as np
import torch

# --- make the three sibling folders importable -----------------------------
SIH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for subfolder in ("", "person2_elevation", "terrain_classifier"):
    path = os.path.join(SIH_ROOT, subfolder) if subfolder else SIH_ROOT
    if path not in sys.path:
        sys.path.insert(0, path)

from depth_anything_v2.dpt import DepthAnythingV2          # noqa: E402  (Person 1)
from terrain_classifier import classify_terrain             # noqa: E402
from elevation_pipeline import generate_elevation            # noqa: E402  (Person 2)

# --- model config (mirrors terrain_classifier/run.py) -----------------------
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
ENCODER = os.getenv("DEPTH_MODEL_ENCODER", "vitl")
CHECKPOINT_PATH = os.path.join(SIH_ROOT, "checkpoints", f"depth_anything_v2_{ENCODER}.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

RESULTS_DIR = os.path.join(SIH_ROOT, "app", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- lazy singleton model load -----------------------------------------------
# Loading the ViT checkpoint takes real time; do it once per process, not
# once per upload. Guarded by a lock since FastAPI's BackgroundTasks can
# run concurrently.
_model = None
_model_lock = threading.Lock()


def _get_model() -> DepthAnythingV2:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            if not os.path.isfile(CHECKPOINT_PATH):
                raise FileNotFoundError(
                    f"Depth model checkpoint not found at {CHECKPOINT_PATH}. "
                    f"Download depth_anything_v2_{ENCODER}.pth from the Depth "
                    f"Anything V2 releases and place it in a `checkpoints/` "
                    f"folder at the repo root."
                )
            model = DepthAnythingV2(**MODEL_CONFIGS[ENCODER])
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
            model = model.to(DEVICE).eval()
            _model = model
    return _model


def run_pipeline(image_path: str, srtm_path: str | None = None) -> dict:
    """
    Runs the full pipeline on one uploaded image:
      1. Load image
      2. Depth Anything V2 -> relative depth map, normalized 0-1
      3. OpenCV terrain classifier -> per-pixel terrain class mask
      4. Elevation calibration -> real-world-scale elevation
         (terrain-aware if srtm_path given, else "relative" mode - a
         plausible but uncalibrated height scale, per elevation_pipeline.py)
      5. Save a height-map visualization PNG

    Returns a dict shaped for the `Result` DB model:
        height_map_path, flythrough_path, min/max/mean_height_m

    `flythrough_path` is always None for now - 3D mesh/flythrough
    generation from the elevation grid isn't built yet (see README).
    """
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise ValueError(f"Could not read image at {image_path} (unsupported or corrupt file)")

    model = _get_model()

    # 1. Depth inference (relative depth, arbitrary scale)
    depth = model.infer_image(raw_image)
    depth = (depth - depth.min()) / (depth.max() - depth.min())  # normalize 0-1, matches person2's format contract

    # 2. Terrain classification (pure OpenCV, no model/checkpoint needed)
    terrain_mask = classify_terrain(raw_image)
    assert terrain_mask.shape == depth.shape, (
        f"terrain_mask shape {terrain_mask.shape} != depth shape {depth.shape} - alignment broken"
    )

    # 3. Elevation calibration (terrain-aware if SRTM given, else relative-scale fallback)
    elevation, metadata = generate_elevation(depth, srtm_path=srtm_path, terrain_mask=terrain_mask)

    # 4. Save a visualized height map
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    height_map_filename = f"{base_name}_heightmap.png"
    height_map_full_path = os.path.join(RESULTS_DIR, height_map_filename)
    _save_height_map_png(elevation, height_map_full_path)

    return {
        "height_map_path": f"/static/results/{height_map_filename}",
        "flythrough_path": None,  # not yet built - see README "Next steps"
        "min_height_m": float(np.nanmin(elevation)),
        "max_height_m": float(np.nanmax(elevation)),
        "mean_height_m": float(np.nanmean(elevation)),
        "calibration_mode": metadata["mode"],
        "error_mean": metadata["error_mean"],
        "error_max": metadata["error_max"],
        "correlation": metadata["correlation"],
        "artifact_pixels": metadata["artifact_pixels"],
        "fit_report": json.dumps(metadata["fit_report"], default=float) if metadata["fit_report"] is not None else None,
    }


def _save_height_map_png(elevation: np.ndarray, out_path: str) -> None:
    """Normalizes an elevation array to 0-255 and writes it as a grayscale PNG."""
    e_min, e_max = np.nanmin(elevation), np.nanmax(elevation)
    span = e_max - e_min
    normalized = np.zeros_like(elevation, dtype=np.uint8) if span == 0 else (
        ((elevation - e_min) / span) * 255
    ).astype(np.uint8)
    cv2.imwrite(out_path, normalized)