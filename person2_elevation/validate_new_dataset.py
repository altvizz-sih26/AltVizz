"""
Honest calibration validator for a new labeled dataset.

Use this whenever you receive a depth map + terrain mask + SRTM/reference
elevation for an image (or a before/after pair) and want to know how
accurate the calibration REALLY is on data it never saw during fitting.

This does NOT touch elevation_pipeline.py or affect production behavior.
It's a separate, honest test harness — reuses fit_methods.py and
metrics.py, the same functions already proven correct on GAMUS.

HOW TO USE:
Just change the four paths in CONFIG below to point at whatever new
files you receive, then run:
    python person2_elevation/validate_new_dataset.py
"""

import numpy as np
import rasterio
from pathlib import Path

from fit_methods import fit_terrain_wise, predict_terrain_wise
from metrics import compute_metrics

# ============ CONFIG — change these four lines per dataset/image ============
DEPTH_PATH = "person2_elevation/bare_terrain_1_depth.npy"
TERRAIN_MASK_PATH = "person2_elevation/bare_terrain_1_terrain_mask.npy"
SRTM_PATH = "person2_elevation/srtm_bare_terrain_aligned.tif"
LABEL = "bare_terrain_1"  # just a name for the printed report, doesn't affect logic
# ==============================================================================

TEST_FRACTION = 0.2
SEED = 42
CLASS_NAMES = {0: "unknown", 1: "urban", 2: "vegetation", 3: "shadow",
               4: "bare_ground", 5: "water"}  # adjust if your class ids differ


def load_reference_elevation(srtm_path, shape):
    with rasterio.open(srtm_path) as ds:
        ref = ds.read(1).astype(np.float32)
    if ref.shape != shape:
        raise ValueError(
            f"SRTM shape {ref.shape} doesn't match depth shape {shape} — "
            "these need to already be aligned/resampled to match before running this."
        )
    return ref


def main():
    depth = np.load(DEPTH_PATH).astype(np.float32)
    terrain_mask = np.load(TERRAIN_MASK_PATH)
    reference_elevation = load_reference_elevation(SRTM_PATH, depth.shape)

    valid = np.isfinite(depth) & np.isfinite(reference_elevation)
    print(f"[{LABEL}] valid pixels: {valid.sum()} / {valid.size}")

    # --- the honest split: decide train/test BEFORE any fitting happens ---
    rng = np.random.default_rng(SEED)
    rand_vals = rng.random(valid.shape)
    is_test = valid & (rand_vals < TEST_FRACTION)
    is_train = valid & ~is_test
    print(f"[{LABEL}] train pixels: {is_train.sum()}, test pixels: {is_test.sum()}")

    depth_train, ref_train, class_train = depth[is_train], reference_elevation[is_train], terrain_mask[is_train]
    depth_test, ref_test, class_test = depth[is_test], reference_elevation[is_test], terrain_mask[is_test]

    # --- fit ONLY on training pixels (this mirrors terrain_aware_calibration's method) ---
    fit_report = fit_terrain_wise(depth_train, ref_train, class_train,
                                   model_type="huber", degree=2)

    # --- predict + score ONLY on held-out test pixels ---
    predicted_test = predict_terrain_wise(depth_test, class_test, fit_report)

    print(f"\n=== HONEST held-out results for {LABEL} (never seen during fitting) ===")
    pooled = compute_metrics(predicted_test, ref_test)
    print(f"Pooled: corr={pooled['correlation']:.3f}  mae={pooled['mae']:.3f}  "
          f"rmse={pooled['rmse']:.3f}  n={pooled['n']}")

    print("\nPer terrain class:")
    for class_id, class_name in CLASS_NAMES.items():
        class_pixel_mask = (class_test == class_id)
        if class_pixel_mask.sum() < 10:
            print(f"  {class_name:12s}: too few test pixels ({class_pixel_mask.sum()}), skipped")
            continue
        m = compute_metrics(predicted_test[class_pixel_mask], ref_test[class_pixel_mask])
        print(f"  {class_name:12s}: corr={m['correlation']:.3f}  mae={m['mae']:.3f}  "
              f"rmse={m['rmse']:.3f}  n={m['n']}")

    print(f"\nThis is the number to report as accuracy — it was never shown to the fit.")


if __name__ == "__main__":
    main()