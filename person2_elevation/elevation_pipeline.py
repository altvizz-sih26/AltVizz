import numpy as np
import rasterio

CLASS_NAMES = {
    0: "urban",
    1: "vegetation",
    2: "bare_terrain",
    3: "water",
    4: "shadow",
    5: "unknown_low_confidence",
}


def load_elevation_any(path):
    """Reads elevation ground truth from either GeoTIFF or .npy."""
    if path.endswith(".tif") or path.endswith(".tiff"):
        with rasterio.open(path) as dataset:
            elevation = dataset.read(1).astype(np.float32)
    elif path.endswith(".npy"):
        elevation = np.load(path).astype(np.float32)
    else:
        raise ValueError(f"Unrecognized file type: {path}")
    void_mask = (elevation == -32768) | (elevation < -1000)
    return elevation, void_mask


def detect_edge_artifacts(elevation, border=5):
    """Flags edge strips that are suspiciously 100% exact 0.0 (reprojection artifacts)."""
    artifact_mask = np.zeros_like(elevation, dtype=bool)
    edges = {
        "top": elevation[:border, :],
        "bottom": elevation[-border:, :],
        "left": elevation[:, :border],
        "right": elevation[:, -border:],
    }
    for name, region in edges.items():
        zero_fraction = (region == 0.0).mean()
        if zero_fraction > 0.5:
            print(f"Edge '{name}': {zero_fraction*100:.0f}% exactly 0.0 — flagging as likely artifact")
            if name == "top": artifact_mask[:border, :] |= (region == 0.0)
            elif name == "bottom": artifact_mask[-border:, :] |= (region == 0.0)
            elif name == "left": artifact_mask[:, :border] |= (region == 0.0)
            elif name == "right": artifact_mask[:, -border:] |= (region == 0.0)
    return artifact_mask


def _resample_to_match(reference_elevation, void_mask, target_shape):
    from scipy.ndimage import zoom
    zoom_factors = (target_shape[0] / reference_elevation.shape[0],
                     target_shape[1] / reference_elevation.shape[1])
    reference_elevation = zoom(reference_elevation, zoom_factors, order=0)
    void_mask = zoom(void_mask.astype(np.uint8), zoom_factors, order=0).astype(bool)
    return reference_elevation, void_mask


def relative_elevation(depth, min_height=0, max_height=50, invert=False):
    """No SRTM available — arbitrary but plausible height scale, no error metric."""
    if invert:
        depth = 1 - depth
    elevation = depth * (max_height - min_height) + min_height
    return elevation.astype(np.float32)


def global_calibration(depth, reference_elevation, void_mask):
    """SRTM available, no terrain mask — one fit for the whole image."""
    valid = ~void_mask
    a, b = np.polyfit(depth[valid].flatten(), reference_elevation[valid].flatten(), 1)
    calibrated = a * depth + b
    return calibrated, {"global": (a, b, int(valid.sum()))}


def terrain_aware_calibration(depth, reference_elevation, void_mask, terrain_mask, min_pixels=500):
    """SRTM + terrain mask available — separate fit per terrain class."""
    calibrated = np.zeros_like(depth)
    valid = ~void_mask
    a_global, b_global = np.polyfit(depth[valid].flatten(), reference_elevation[valid].flatten(), 1)

    fit_report = {}
    for cls in np.unique(terrain_mask):
        class_pixels = (terrain_mask == cls) & valid
        n_pixels = int(class_pixels.sum())
        if n_pixels < min_pixels:
            a, b = a_global, b_global
        else:
            a, b = np.polyfit(depth[class_pixels].flatten(), reference_elevation[class_pixels].flatten(), 1)
        fit_report[int(cls)] = (a, b, n_pixels)
        calibrated[terrain_mask == cls] = a * depth[terrain_mask == cls] + b

    return calibrated, fit_report


def generate_elevation(depth, srtm_path=None, terrain_mask=None):
    """
    Main entry point. Takes already-loaded depth array, an optional path
    to SRTM ground truth, and an optional already-loaded terrain mask.
    Automatically picks relative / global / terrain-aware calibration.
    Automatically excludes detected edge artifacts from error reporting.
    """
    metadata = {"error_mean": None, "error_max": None, "correlation": None,
                "fit_report": None, "mode": None, "artifact_pixels": 0}

    if srtm_path is None:
        metadata["mode"] = "relative"
        elevation = relative_elevation(depth)
        return elevation, metadata

    reference_elevation, void_mask = load_elevation_any(srtm_path)

    if depth.shape != reference_elevation.shape:
        print(f"WARNING: shape mismatch — depth {depth.shape} vs reference {reference_elevation.shape}. Resampling.")
        reference_elevation, void_mask = _resample_to_match(reference_elevation, void_mask, depth.shape)

    artifact_mask = detect_edge_artifacts(reference_elevation)
    exclude_mask = void_mask | artifact_mask
    metadata["artifact_pixels"] = int(artifact_mask.sum())

    if terrain_mask is not None:
        if depth.shape != terrain_mask.shape:
            raise ValueError(f"Depth {depth.shape} and terrain_mask {terrain_mask.shape} must match.")
        metadata["mode"] = "terrain_aware"
        elevation, fit_report = terrain_aware_calibration(depth, reference_elevation, exclude_mask, terrain_mask)
    else:
        metadata["mode"] = "global"
        elevation, fit_report = global_calibration(depth, reference_elevation, exclude_mask)

    valid = ~exclude_mask
    error = np.abs(elevation[valid] - reference_elevation[valid])
    metadata["error_mean"] = float(error.mean())
    metadata["error_max"] = float(error.max())
    metadata["correlation"] = float(np.corrcoef(elevation[valid].flatten(), reference_elevation[valid].flatten())[0, 1])
    metadata["fit_report"] = fit_report

    return elevation, metadata