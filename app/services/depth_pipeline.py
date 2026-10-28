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
from pyproj import Geod
from rasterio.warp import Resampling, reproject, transform as transform_coordinates, transform_bounds

# --- make the sibling folders importable -----------------------------
SIH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for subfolder in ("", "person2_elevation", "terrain_classifier", "person3_3d_reconstruction"):
    path = os.path.join(SIH_ROOT, subfolder) if subfolder else SIH_ROOT
    if path not in sys.path:
        sys.path.insert(0, path)

from gamus_finetune.model import build_gamus_model        # noqa: E402  (Person 1)
from terrain_classifier import classify_terrain             # noqa: E402
from elevation_pipeline import CLASS_NAMES, generate_elevation, load_elevation_any, detect_edge_artifacts  # noqa: E402  (Person 2)
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

FINETUNED_PATH = os.path.join(
    SIH_ROOT, "gamus_finetune", "runs", "gamus_vits_run1", "best.pt"
)
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
            model = build_gamus_model(ENCODER, CHECKPOINT_PATH, device=DEVICE)
            ckpt = torch.load(FINETUNED_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state"])
            model = model.eval()
            print("Loaded model:", type(model).__name__, "epoch", ckpt["epoch"])
            _model = model
    return _model


def _auto_fetch_srtm(
    image_path: str,
    center_lat: float | None = None,
    center_lon: float | None = None,
    ground_width_m: float | None = None,
    exaggeration: float | None = None,
) -> tuple[str | None, str]:
    """
    Tries to automatically get a matching SRTM reference for this upload
    using Person 5's generic process_new_image.py, instead of requiring the
    caller to pass srtm_path manually.

    Returns the aligned SRTM path and a user-visible status.
    """
    try:
        path = process_new_image(
            image_path,
            output_dir=RESULTS_DIR,
            center_lat=center_lat,
            center_lon=center_lon,
            ground_width_m=ground_width_m,
        )
        return path, "SRTM reference downloaded and aligned"
    except Exception as e:
        reason = f"SRTM unavailable: {e}"
        _key = os.getenv("OPENTOPO_API_KEY", "").strip()
        if _key:
            reason = reason.replace(_key, "[hidden]")
        if "API_Key=" in reason:
            head, _, tail = reason.partition("API_Key=")
            reason = head + "API_Key=[hidden]" + tail[len(tail.split("&")[0].split(" ")[0].split("'")[0]):]
        _logger.warning("%s (input %s)", reason, image_path)
        return None, reason


def _pixel_size_m(
    image_path: str,
    ground_width_m: float | None = None,
) -> tuple[float, float] | None:
    with rasterio.open(image_path) as source:
        if source.crs is None:
            if ground_width_m is not None:
                pixel_width = ground_width_m / source.width
                return float(pixel_width), float(pixel_width)
            return None

        col, row = source.width / 2, source.height / 2
        x0, y0 = source.transform * (col, row)
        x1, y1 = source.transform * (col + 1, row)
        x2, y2 = source.transform * (col, row + 1)

        if source.crs.is_projected:
            unit_to_m = source.crs.linear_units_factor[1]
            return (
                float(np.hypot(x1 - x0, y1 - y0) * unit_to_m),
                float(np.hypot(x2 - x0, y2 - y0) * unit_to_m),
            )

        xs, ys = transform_coordinates(
            source.crs,
            "EPSG:4326",
            [x0, x1, x2],
            [y0, y1, y2],
        )
        geod = Geod(ellps="WGS84")
        width_m = abs(geod.inv(xs[0], ys[0], xs[1], ys[1])[2])
        height_m = abs(geod.inv(xs[0], ys[0], xs[2], ys[2])[2])
        return float(width_m), float(height_m)

def _geo_corners(image_path: str):
    """Lon/lat of the four corner pixel centres (top-left, top-right, bottom-left, bottom-right)."""
    try:
        from rasterio.warp import transform as _warp_transform
        with rasterio.open(image_path) as ds:
            if ds.crs is None or ds.transform.is_identity:
                return None
            h, w = ds.height, ds.width
            pts = [ds.xy(0, 0), ds.xy(0, w - 1), ds.xy(h - 1, 0), ds.xy(h - 1, w - 1)]
            lons, lats = _warp_transform(ds.crs, "EPSG:4326", [p[0] for p in pts], [p[1] for p in pts])
            return [float(v) for lon, lat in zip(lons, lats) for v in (lon, lat)]
    except Exception:
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


def _try_generate_glb(
    elevation: np.ndarray,
    image_path: str,
    base_name: str,
    pixel_size_m: tuple[float, float] | None = None,
    exaggeration: float = 1.0,
) -> str:
    """
    Builds the textured 3D mesh (Person 3's create_terrain.py, now a
    reusable function) from the elevation this pipeline just computed.

    Raises on failure so the job cannot be marked complete without its 3D model.
    """
    texture_path = None
    try:
        glb_filename = f"{base_name}_terrain.glb"
        glb_full_path = os.path.join(RESULTS_DIR, glb_filename)
        texture_image_path, texture_path = _prepare_glb_texture(image_path, base_name)
        generate_glb(
            elevation,
            texture_image_path,
            glb_full_path,
            pixel_size_m=pixel_size_m,
            vertical_exaggeration=exaggeration,
        )
        return f"/static/results/{glb_filename}"
    except Exception as e:
        _logger.exception("GLB mesh generation failed for %s", image_path)
        raise RuntimeError(f"3D mesh generation failed: {e}") from e
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
    mode: str | None = None,
    srtm_status: str | None = None,
    pixel_size_m: tuple | None = None,
    spread_metrics: dict | None = None,
    geo_corners: list | None = None,

) -> str:
    """
    NEW: Saves the elevation grid as <base_name>_analysis.npz next to the GLB,
    so the 3D viewer can build the Analysis Map, slope and region statistics
    for this upload instead of showing "No analysis data".

    The array is stored under the key `elevation`, which is what
    normalizeGrid() in src/main.js reads for a single-snapshot file. That
    function defaults to `elevation_before`, so saving only
    `elevation_after` would leave the maps empty. A single upload has no
    "before" snapshot.

    Raises on failure so a completed job always has its analysis file.
    """
    try:
        elev = np.asarray(elevation, dtype=np.float32)
        original_width = elev.shape[1]

        # Downscale very large grids so the browser download stays small.
        longest = max(elev.shape[:2])
        if longest > ANALYSIS_MAX_SIDE:
            scale = ANALYSIS_MAX_SIDE / longest
            new_w = max(1, int(round(elev.shape[1] * scale)))
            new_h = max(1, int(round(elev.shape[0] * scale)))
            elev = cv2.resize(elev, (new_w, new_h), interpolation=cv2.INTER_AREA)

        arrays = {"elevation": elev}
        if pixel_size_m:
            arrays["pixel_size_m"] = np.float32(sum(pixel_size_m) / 2 * original_width / elev.shape[1])
        for name, values in (("terrain", terrain_mask), ("confidence", confidence)):
            if geo_corners:
                arrays["geo_corners"] = np.array(geo_corners, dtype=np.float64)
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
        if mode is not None:
            arrays["mode"] = np.array(mode)
        if srtm_status is not None:
            arrays["srtm_status"] = np.array(srtm_status)
            if spread_metrics is not None:
                arrays["metrics_after"] = np.array(json.dumps({"pooled": spread_metrics}))


        npz_filename = f"{base_name}_analysis.npz"
        np.savez_compressed(os.path.join(RESULTS_DIR, npz_filename), **arrays)
        return f"/static/results/{npz_filename}"
    except Exception as e:
        _logger.exception("Analysis NPZ generation failed for %s", base_name)
        raise RuntimeError(f"Analysis file generation failed: {e}") from e


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


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        raise ValueError("Depth model returned no finite values")
    lower, upper = np.percentile(depth[finite], [1, 99])
    if upper <= lower:
        return np.zeros_like(depth, dtype=np.float32)
    return np.clip((depth - lower) / (upper - lower), 0, 1).astype(np.float32)


def reconstruct(
    image_path: str,
    srtm_path: str | None = None,
    progress=lambda stage, pct: None,
    center_lat: float | None = None,
    center_lon: float | None = None,
    ground_width_m: float | None = None,
    exaggeration: float | None = None,
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
    if srtm_path is not None:
        srtm_status = "SRTM reference file supplied"
    else:
        srtm_path, srtm_status = _auto_fetch_srtm(
            image_path,
            center_lat=center_lat,
            center_lon=center_lon,
            ground_width_m=ground_width_m,
        )
    _logger.info("SRTM Reference completed in %.3f seconds", time.perf_counter() - started)

    started = time.perf_counter()
    progress("Depth Estimation", 15)
    model = _get_model()
    tensor, (h, w) = model.backbone_and_head.image2tensor(raw_image, 518)
    with torch.no_grad():
        height_m = torch.nn.functional.interpolate(model(tensor.to(DEVICE)), (h, w), mode="bilinear", align_corners=True)[0, 0].cpu().numpy()
        depth = _normalize_depth(height_m)
        flipped = np.ascontiguousarray(raw_image[:, ::-1])
    tensor_f, (hf, wf) = model.backbone_and_head.image2tensor(flipped, 518)
    with torch.no_grad():
        height_flip = torch.nn.functional.interpolate(model(tensor_f.to(DEVICE)), (hf, wf), mode="bilinear", align_corners=True)[0, 0].cpu().numpy()[:, ::-1]
    disagreement = np.abs(height_m - height_flip)
    spread = {
        "rmse": float(np.sqrt(np.mean((height_m - height_flip) ** 2))),
        "mae": float(disagreement.mean()),
        "correlation": float(np.corrcoef(height_m.ravel(), height_flip.ravel())[0, 1]),
    }
    uncertainty = cv2.GaussianBlur(disagreement.astype(np.float32), (0, 0), 3)
    
    
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
    elevation = height_m
    metadata = {"mode": "model_metric", "error_mean": None, "error_max": None,
            "correlation": None, "artifact_pixels": 0, "fit_report": None}
    if srtm_path is not None:
        ground, void = load_elevation_any(srtm_path)
        if ground.shape != height_m.shape:
            ground = cv2.resize(ground.astype(np.float32), (height_m.shape[1], height_m.shape[0]), interpolation=cv2.INTER_LINEAR)
            void = cv2.resize(void.astype(np.uint8), (height_m.shape[1], height_m.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        bad = void | detect_edge_artifacts(ground)
        if (~bad).any():
            ground = np.where(bad, np.median(ground[~bad]), ground)
            elevation = (ground + height_m).astype(np.float32)
            metadata["mode"] = "model_plus_srtm"
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
    px = _pixel_size_m(image_path, ground_width_m)
    if exaggeration is None:
        exaggeration = 1.0
        relief = float(np.nanpercentile(elevation, 98) - np.nanpercentile(elevation, 2))
        if px and relief > 0:
            exaggeration = float(np.clip(0.12 * px[0] * elevation.shape[1] / relief, 1.0, 8.0))
    glb_url = _try_generate_glb(
        elevation,
        image_path,
        base_name,
        px,
        exaggeration,
    )
    _logger.info("Building 3D Mesh completed in %.3f seconds", time.perf_counter() - started)

    return {
        "elevation": elevation,
        "depth": depth,
        "height_m": height_m,
        "terrain_mask": terrain_mask,
        "meta": metadata,
        "base_name": base_name,
        "srtm_path": srtm_path,
        "pixel_size_m": _pixel_size_m(image_path, ground_width_m),
        "spread": spread,
        "uncertainty": uncertainty,
        "srtm_status": srtm_status,
        "heightmap_url": heightmap_url,
        "dsm_geotiff_url": dsm_geotiff_url,
        "glb_url": glb_url,
        "exaggeration": exaggeration,
        "geo_corners": _geo_corners(image_path),
    }


def run_pipeline(
    image_path: str,
    srtm_path: str | None = None,
    progress=lambda stage, pct: None,
    center_lat: float | None = None,
    center_lon: float | None = None,
    ground_width_m: float | None = None,
) -> dict:
    """Runs reconstruction and returns the existing Result-model fields.

    The analysis NPZ continues to store its grid under the `elevation` key.
    """
    result = reconstruct(
        image_path,
        srtm_path,
        progress,
        center_lat=center_lat,
        center_lon=center_lon,
        ground_width_m=ground_width_m,
    )
    elevation = result["elevation"]
    metadata = result["meta"]

    # Save the analysis grid under the key expected by the viewer.
    analysis_path = _save_analysis_npz(
        elevation,
        result["base_name"],
        terrain_mask=result["terrain_mask"],
        confidence=(1.0 / (1.0 + result["uncertainty"] / 2.0)).astype(np.float32),
        mode=metadata["mode"],
        srtm_status=result["srtm_status"],
        pixel_size_m=result["pixel_size_m"],
        spread_metrics=result["spread"],
        geo_corners=result["geo_corners"],

    )

    fit_report = metadata["fit_report"]
    if isinstance(fit_report, dict):
        fit_report = {**fit_report, "srtm_status": result["srtm_status"]}
    else:
        fit_report = {
            "calibration": fit_report,
            "srtm_status": result["srtm_status"],
        }
    fit_report["spread"] = result["spread"]

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
        "fit_report": json.dumps(fit_report, default=float),
    }


def run_compare_pipeline(
    before_path: str,
    after_path: str,
    srtm_path: str | None = None,
    progress=lambda stage, pct: None,
    center_lat: float | None = None,
    center_lon: float | None = None,
    ground_width_m: float | None = None,
) -> dict:
    """Reconstructs and compares a before/after image pair."""
    def before_progress(stage, pct):
        progress(f"Before: {stage}", round(pct * 0.4))

    def after_progress(stage, pct):
        progress(f"After: {stage}", 45 + round(pct * 0.4))

    before = reconstruct(
        before_path,
        srtm_path,
        before_progress,
        center_lat=center_lat,
        center_lon=center_lon,
        ground_width_m=ground_width_m,
    )
    progress("Aligning After Image", 42)
    aligned_after_path = align_after_to_before(before_path, after_path)
    after = reconstruct(aligned_after_path, before["srtm_path"], after_progress, exaggeration=before["exaggeration"])

    use_srtm = before["srtm_path"] is not None
    progress("Comparing Elevation", 90)
    comparison = run_before_during(
        before["elevation"],
        after["elevation"],
        before["terrain_mask"],
        after["terrain_mask"],
        srtm_path=before["srtm_path"],
        use_srtm=False,
        conf_before=(1.0 / (1.0 + before["uncertainty"] / 2.0)).astype(np.float32),
        conf_after=(1.0 / (1.0 + after["uncertainty"] / 2.0)).astype(np.float32),
    )

    comparison_name = f"{before['base_name']}_compare"
    progress("Saving Comparison", 95)
    save_for_person4(comparison, comparison_name, out_dir=RESULTS_DIR,pixel_size_m=before["pixel_size_m"], geo_corners=before["geo_corners"])
    change_summary = summarize_difference_grid(comparison["diff_grid"])
    change_summary["srtm_status"] = before["srtm_status"]
    progress("Comparison Complete", 100)

    return {
        "analysis_path": f"/static/results/{comparison_name}.npz",
        "change_summary": change_summary,
        "mode": comparison["mode"],
        "srtm_status": before["srtm_status"],
        "calibration_mode": "model_metric",
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