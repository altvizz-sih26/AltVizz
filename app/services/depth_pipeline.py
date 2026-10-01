"""
Real height-estimation pipeline: Depth Anything V2 -> terrain classification
-> elevation calibration.

Replaces the fake random output in ml_stub.py. Wires together four
previously-separate pieces of the project:

  1. depth_anything_v2/           (Person 1) - monocular depth estimation
  2. terrain_classifier/          - OpenCV terrain classification (no ML,
                                    no checkpoint needed)
  3. person2_elevation/           (Person 2) - depth -> real-world elevation
                                    calibration, optionally against SRTM
  4. process_new_image.py         (Person 5) - auto-fetches + aligns SRTM
                                    reference elevation for ANY georeferenced
                                    upload, no hardcoded region needed

None of these three (four) live in a proper installable package today, so this
module adds their folders to sys.path once at import time and then imports
them normally. If the project structure changes, only SIH_ROOT below needs
updating.

REQUIRES: checkpoints/depth_anything_v2_<encoder>.pth to exist at the repo
root. The team's checkpoint is the Small model, so the file is
checkpoints/depth_anything_v2_vits.pth and the default encoder is "vits".
Override with the DEPTH_MODEL_ENCODER env var if you use a different one.

OUTPUTS per upload (all written to app/results/ and served under
/static/results/):
  <name>_heightmap.png   - grayscale height map
  <name>_terrain.glb     - textured 3D mesh
  <name>_analysis.npz    - elevation grid for the viewer's Analysis Map,
                           slope and region stats. The frontend derives its
                           URL from the GLB URL (_terrain.glb -> _analysis.npz),
                           so no database change is needed.
"""
import json
import logging
import os
import sys
import threading
import time

import cv2
import numpy as np
import rasterio
import torch
from rasterio.warp import Resampling, reproject, transform_bounds

# --- make the sibling folders importable -----------------------------
SIH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for subfolder in ("", "person2_elevation", "terrain_classifier", "person3_3d_reconstruction"):
    path = os.path.join(SIH_ROOT, subfolder) if subfolder else SIH_ROOT
    if path not in sys.path:
        sys.path.insert(0, path)

from depth_anything_v2.dpt import DepthAnythingV2          # noqa: E402  (Person 1)
from terrain_classifier import classify_terrain             # noqa: E402
from elevation_pipeline import CLASS_NAMES, generate_elevation  # noqa: E402  (Person 2)
from confidence import compute_confidence                     # noqa: E402
from process_new_image import process_new_image             # noqa: E402  (Person 5)
from create_terrain import generate_glb                      # noqa: E402  (Person 3)
from run_before_during_pipeline import run_before_during, save_for_person4  # noqa: E402
from diff_grid import summarize_difference_grid               # noqa: E402

# --- model config (mirrors terrain_classifier/run.py) -----------------------
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
ENCODER = os.getenv("DEPTH_MODEL_ENCODER", "vits")
CHECKPOINT_PATH = os.path.abspath(os.getenv(
    "DEPTH_MODEL_CHECKPOINT",
    os.path.join(SIH_ROOT, "checkpoints", f"depth_anything_v2_{ENCODER}.pth"),
))
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

RESULTS_DIR = os.path.join(SIH_ROOT, "app", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Largest side (in pixels) of the elevation grid saved for the viewer.
# Keeps the .npz small enough to download quickly in the browser.
ANALYSIS_MAX_SIDE = 1024
_logger = logging.getLogger(__name__)

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
                    f"Expected depth_anything_v2_{ENCODER}.pth inside the "
                    f"`checkpoints/` folder at the repo root."
                )
            model = DepthAnythingV2(**MODEL_CONFIGS[ENCODER])
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
            model = model.to(DEVICE).eval()
            _model = model
    return _model


def _auto_fetch_srtm(image_path: str) -> str | None:
    """
    Tries to automatically get a matching SRTM reference for this upload
    using Person 5's generic process_new_image.py, instead of requiring the
    caller to pass srtm_path manually.

    Returns the aligned SRTM path on success, or None if it can't be fetched
    (e.g. the image has no georeferencing, or the SRTM download fails/network
    is unavailable) - falling back to relative/uncalibrated mode, same as
    before, rather than crashing the whole pipeline.
    """
    try:
        return process_new_image(image_path)
    except Exception as e:
        print(f"[depth_pipeline] SRTM auto-fetch skipped for {image_path}: {e}")
        return None


def _prepare_glb_texture(image_path: str, base_name: str) -> tuple[str, str | None]:
    """Convert TIFF textures through GDAL so LZW inputs don't need imagecodecs."""
    if not image_path.lower().endswith((".tif", ".tiff")):
        return image_path, None

    with rasterio.open(image_path) as source:
        band_indexes = [min(index, source.count) for index in (1, 2, 3)]
        rgb = np.moveaxis(source.read(band_indexes), 0, -1)

    rgb = np.nan_to_num(rgb.astype(np.float32), nan=0, posinf=255, neginf=0)
    max_value = float(rgb.max())
    if max_value > 255:
        rgb *= 255 / max_value
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    texture_path = os.path.join(RESULTS_DIR, f"{base_name}_texture.png")
    if not cv2.imwrite(texture_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write temporary mesh texture to {texture_path}")
    return texture_path, texture_path


def _try_generate_glb(elevation: np.ndarray, image_path: str, base_name: str) -> str | None:
    """
    Builds the textured 3D mesh (Person 3's create_terrain.py, now a
    reusable function) from the elevation this pipeline just computed.

    Returns the GLB's public path on success, or None if mesh generation
    fails for any reason (e.g. bad image format) - the rest of the result
    (heightmap, DSM stats) still gets returned either way, same pattern as
    the SRTM auto-fetch fallback above.
    """
    texture_path = None
    try:
        glb_filename = f"{base_name}_terrain.glb"
        glb_full_path = os.path.join(RESULTS_DIR, glb_filename)
        texture_image_path, texture_path = _prepare_glb_texture(image_path, base_name)
        generate_glb(elevation, texture_image_path, glb_full_path)
        return f"/static/results/{glb_filename}"
    except Exception as e:
        print(f"[depth_pipeline] GLB mesh generation failed for {image_path}: {e}")
        return None
    finally:
        if texture_path and os.path.exists(texture_path):
            os.remove(texture_path)


def align_after_to_before(before_path: str, after_path: str) -> str:
    """Aligns the after image to the before image's pixel grid.

    Georeferenced images are reprojected to the before image's CRS,
    transform, and dimensions. Otherwise, ECC registration estimates a
    homography on the before/after images and warps the after image onto the
    before image's pixel grid.
    """
    before_name = os.path.splitext(os.path.basename(before_path))[0]
    after_name = os.path.splitext(os.path.basename(after_path))[0]
    aligned_path = os.path.join(RESULTS_DIR, f"{after_name}_aligned_to_{before_name}.tif")

    try:
        with rasterio.open(before_path) as before, rasterio.open(after_path) as after:
            if before.crs is None and after.crs is None:
                before_shape = (before.height, before.width)
                output_path = os.path.splitext(aligned_path)[0] + ".png"
            elif before.crs is None or after.crs is None:
                raise ValueError("Both images must be georeferenced, or neither, to align them")
            else:
                after_bounds = transform_bounds(after.crs, before.crs, *after.bounds)
                if (
                    after_bounds[2] <= before.bounds.left
                    or after_bounds[0] >= before.bounds.right
                    or after_bounds[3] <= before.bounds.bottom
                    or after_bounds[1] >= before.bounds.top
                ):
                    raise ValueError("Before and after images do not overlap geographically")

                band_indexes = [min(index + 1, after.count) for index in range(3)]
                output = np.zeros(
                    (3, before.height, before.width),
                    dtype=np.dtype(after.dtypes[0]),
                )
                for output_band, source_band in enumerate(band_indexes):
                    reproject(
                        source=rasterio.band(after, source_band),
                        destination=output[output_band],
                        src_transform=after.transform,
                        src_crs=after.crs,
                        dst_transform=before.transform,
                        dst_crs=before.crs,
                        resampling=Resampling.bilinear,
                    )

                profile = before.profile.copy()
                profile.update(
                    driver="GTiff",
                    count=3,
                    dtype=output.dtype,
                    compress="lzw",
                )
                with rasterio.open(aligned_path, "w", **profile) as destination:
                    destination.write(output)
                return aligned_path

    except rasterio.errors.RasterioIOError:
        before_image = cv2.imread(before_path)
        after_image = cv2.imread(after_path)
        if before_image is None or after_image is None:
            raise ValueError("Could not read both images to align them")
        before_shape = before_image.shape[:2]
        output_path = os.path.splitext(aligned_path)[0] + ".png"

    after_image = cv2.imread(after_path)
    if after_image is None:
        raise ValueError(f"Could not read after image at {after_path}")
    before_image = cv2.imread(before_path)
    if before_image is None:
        raise ValueError(f"Could not read before image at {before_path}")
    before_shape = before_image.shape[:2]
    after_image = cv2.resize(
        after_image,
        (before_shape[1], before_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    before_gray = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    after_gray = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    warp = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        cv2.findTransformECC(
            before_gray,
            after_gray,
            warp,
            cv2.MOTION_HOMOGRAPHY,
            criteria,
        )
    except cv2.error as error:
        raise ValueError(f"Could not register before/after images: {error}") from error
    aligned_image = cv2.warpPerspective(
        after_image,
        warp,
        (before_shape[1], before_shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )
    if not cv2.imwrite(output_path, aligned_image):
        raise OSError(f"Could not write aligned image to {output_path}")
    return output_path


def _save_analysis_npz(
    elevation: np.ndarray,
    base_name: str,
    terrain_mask: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
) -> str | None:
    """
    NEW: Saves the elevation grid as <base_name>_analysis.npz next to the GLB,
    so the 3D viewer can build the Analysis Map, slope and region statistics
    for this upload instead of showing "No analysis data".

    The array is stored under the key `elevation`, which is what
    normalizeGrid() in src/main.js reads for a single-snapshot file. That
    function defaults to `elevation_before`, so saving only
    `elevation_after` would leave the maps empty. A single upload has no
    "before" snapshot.

    Never raises: if saving fails, the rest of the pipeline result is still
    returned and the viewer simply shows the mesh without analytics.
    Returns the public path on success, or None on failure.
    """
    try:
        elev = np.asarray(elevation, dtype=np.float32)

        # Downscale very large grids so the browser download stays small.
        longest = max(elev.shape[:2])
        if longest > ANALYSIS_MAX_SIDE:
            scale = ANALYSIS_MAX_SIDE / longest
            new_w = max(1, int(round(elev.shape[1] * scale)))
            new_h = max(1, int(round(elev.shape[0] * scale)))
            elev = cv2.resize(elev, (new_w, new_h), interpolation=cv2.INTER_AREA)

        arrays = {"elevation": elev}
        for name, values in (("terrain", terrain_mask), ("confidence", confidence)):
            if values is None:
                continue
            values = np.asarray(values)
            if values.shape != elev.shape:
                values = cv2.resize(
                    values,
                    (elev.shape[1], elev.shape[0]),
                    interpolation=cv2.INTER_NEAREST if name == "terrain" else cv2.INTER_AREA,
                )
            arrays[name] = values

        npz_filename = f"{base_name}_analysis.npz"
        np.savez_compressed(os.path.join(RESULTS_DIR, npz_filename), **arrays)
        return f"/static/results/{npz_filename}"
    except Exception as e:
        print(f"[depth_pipeline] analysis npz failed for {base_name}: {e}")
        return None


def _save_dsm_geotiff(elevation: np.ndarray, image_path: str, base_name: str) -> str | None:
    """Writes the elevation grid with source georeferencing when available."""
    with rasterio.open(image_path) as source:
        if source.crs is None:
            return None
        values = np.asarray(elevation, dtype=np.float32)
        if values.shape != (source.height, source.width):
            values = cv2.resize(
                values,
                (source.width, source.height),
                interpolation=cv2.INTER_LINEAR,
            )
        filename = f"{base_name}_dsm.tif"
        output_path = os.path.join(RESULTS_DIR, filename)
        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=np.nan,
            compress="lzw",
        )
        with rasterio.open(output_path, "w", **profile) as destination:
            destination.write(values, 1)
    return f"/static/results/{filename}"


def reconstruct(
    image_path: str,
    srtm_path: str | None = None,
    progress=lambda stage, pct: None,
) -> dict:
    """
    Reconstructs elevation and 3D artifacts for one uploaded image:
      1. Load image
      2. Depth Anything V2 -> relative depth map, normalized 0-1
      3. OpenCV terrain classifier -> per-pixel terrain class mask
      4. If srtm_path wasn't explicitly provided, try to auto-fetch it for
         this specific image (works for any georeferenced upload, not just
         the 3 known test regions)
      5. Elevation calibration -> real-world-scale elevation
         (terrain-aware if srtm_path given/found, else "relative" mode - a
         plausible but uncalibrated height scale, per elevation_pipeline.py)
      6. Save a height-map visualization PNG
      7. Build a textured 3D mesh (.glb) from the elevation + source image

    Returns the computed arrays, metadata, and public artifact URLs.
    `progress` receives a stage name and percentage before each stage.
    Stage durations are written to the module logger.
    """
    started = time.perf_counter()
    progress("Loading Image", 5)
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise ValueError(f"Could not read image at {image_path} (unsupported or corrupt file)")
    _logger.info("Loading Image completed in %.3f seconds", time.perf_counter() - started)

    started = time.perf_counter()
    progress("SRTM Reference", 10)
    if srtm_path is None:
        srtm_path = _auto_fetch_srtm(image_path)
    _logger.info("SRTM Reference completed in %.3f seconds", time.perf_counter() - started)

    started = time.perf_counter()
    progress("Depth Estimation", 15)
    model = _get_model()
    depth = model.infer_image(raw_image)
    depth = (depth - depth.min()) / (depth.max() - depth.min())  # normalize 0-1, matches person2's format contract
    _logger.info("Depth Estimation completed in %.3f seconds", time.perf_counter() - started)

    started = time.perf_counter()
    progress("Terrain Classification", 35)
    terrain_mask = classify_terrain(raw_image)
    assert terrain_mask.shape == depth.shape, (
        f"terrain_mask shape {terrain_mask.shape} != depth shape {depth.shape} - alignment broken"
    )
    _logger.info("Terrain Classification completed in %.3f seconds", time.perf_counter() - started)

    started = time.perf_counter()
    progress("Elevation Calibration", 55)
    elevation, metadata = generate_elevation(depth, srtm_path=srtm_path, terrain_mask=terrain_mask)
    _logger.info("Elevation Calibration completed in %.3f seconds", time.perf_counter() - started)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    started = time.perf_counter()
    progress("Saving Heightmap", 75)
    dsm_geotiff_url = _save_dsm_geotiff(elevation, image_path, base_name)
    height_map_filename = f"{base_name}_heightmap.png"
    height_map_full_path = os.path.join(RESULTS_DIR, height_map_filename)
    _save_height_map_png(elevation, height_map_full_path)
    heightmap_url = f"/static/results/{height_map_filename}"
    _logger.info("Saving Heightmap completed in %.3f seconds", time.perf_counter() - started)

    started = time.perf_counter()
    progress("Building 3D Mesh", 90)
    glb_url = _try_generate_glb(elevation, image_path, base_name)
    _logger.info("Building 3D Mesh completed in %.3f seconds", time.perf_counter() - started)

    return {
        "elevation": elevation,
        "depth": depth,
        "terrain_mask": terrain_mask,
        "meta": metadata,
        "base_name": base_name,
        "srtm_path": srtm_path,
        "heightmap_url": heightmap_url,
        "dsm_geotiff_url": dsm_geotiff_url,
        "glb_url": glb_url,
    }


def run_pipeline(
    image_path: str,
    srtm_path: str | None = None,
    progress=lambda stage, pct: None,
) -> dict:
    """Runs reconstruction and returns the existing Result-model fields.

    The analysis NPZ continues to store its grid under the `elevation` key.
    """
    result = reconstruct(image_path, srtm_path, progress)
    elevation = result["elevation"]
    metadata = result["meta"]

    # Save the analysis grid under the key expected by the viewer.
    analysis_path = _save_analysis_npz(
        elevation,
        result["base_name"],
        terrain_mask=result["terrain_mask"],
        confidence=compute_confidence(result["terrain_mask"], class_names=CLASS_NAMES),
    )

    return {
        "height_map_path": result["heightmap_url"],
        "dsm_geotiff_path": result["dsm_geotiff_url"],
        "analysis_path": analysis_path,
        "flythrough_path": result["glb_url"],
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


def run_compare_pipeline(
    before_path: str,
    after_path: str,
    srtm_path: str | None = None,
    progress=lambda stage, pct: None,
) -> dict:
    """Reconstructs and compares a before/after image pair."""
    def before_progress(stage, pct):
        progress(f"Before: {stage}", round(pct * 0.4))

    def after_progress(stage, pct):
        progress(f"After: {stage}", 45 + round(pct * 0.4))

    before = reconstruct(before_path, srtm_path, before_progress)
    progress("Aligning After Image", 42)
    aligned_after_path = align_after_to_before(before_path, after_path)
    after = reconstruct(aligned_after_path, before["srtm_path"], after_progress)

    use_srtm = before["srtm_path"] is not None
    progress("Comparing Elevation", 90)
    comparison = run_before_during(
        before["depth"],
        after["depth"],
        before["terrain_mask"],
        after["terrain_mask"],
        srtm_path=before["srtm_path"],
        use_srtm=use_srtm,
    )

    comparison_name = f"{before['base_name']}_compare"
    progress("Saving Comparison", 95)
    save_for_person4(comparison, comparison_name, out_dir=RESULTS_DIR)
    change_summary = summarize_difference_grid(comparison["diff_grid"])
    progress("Comparison Complete", 100)

    return {
        "analysis_path": f"/static/results/{comparison_name}.npz",
        "change_summary": change_summary,
        "mode": comparison["mode"],
        "calibration_mode": "srtm_calibrated" if use_srtm else "relative",
        "metrics_before": comparison["metrics_before"],
        "metrics_after": comparison["metrics_after"],
        "before_height_map_path": before["heightmap_url"],
        "before_dsm_geotiff_path": before["dsm_geotiff_url"],
        "before_flythrough_path": before["glb_url"],
        "after_height_map_path": after["heightmap_url"],
        "after_dsm_geotiff_path": after["dsm_geotiff_url"],
        "after_flythrough_path": after["glb_url"],
    }


def _save_height_map_png(elevation: np.ndarray, out_path: str) -> None:
    """Normalizes an elevation array to 0-255 and writes it as a grayscale PNG."""
    e_min, e_max = np.nanmin(elevation), np.nanmax(elevation)
    span = e_max - e_min
    normalized = np.zeros_like(elevation, dtype=np.uint8) if span == 0 else (
        ((elevation - e_min) / span) * 255
    ).astype(np.uint8)
    cv2.imwrite(out_path, normalized)