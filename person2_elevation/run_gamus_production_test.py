import h5py
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from elevation_pipeline import terrain_aware_calibration


# ============================================================
# PATHS
# ============================================================

DEPTH_DIR = Path("person2_elevation/GAMUS_depth")

GAMUS_HEIGHT_DIR = Path("gamus_test/heights")
GAMUS_CLASS_DIR = Path("gamus_test/classes")

OUTPUT_DIR = Path("person2_elevation/GAMUS_production_test_outputs")


# ============================================================
# SETTINGS
# ============================================================

POLYNOMIAL_DEGREE = 2


# GAMUS classes
CLASS_NAMES = {
    0: "Others / background",
    1: "Ground",
    2: "Low vegetation",
    3: "Buildings",
    4: "Water",
    5: "Road",
    6: "Tree",
}


# ============================================================
# LOAD GAMUS FILE
# ============================================================

def load_h5_image(path):
    with h5py.File(path, "r") as f:
        return f["image"][()]


# ============================================================
# PROCESS ONE SAMPLE
# ============================================================

def process_sample(depth_path):

    sample_id = depth_path.stem.replace("_RGB_depth", "")

    height_path = GAMUS_HEIGHT_DIR / f"{sample_id}_AGL.h5"
    class_path = GAMUS_CLASS_DIR / f"{sample_id}_CLS.h5"

    print("\n" + "=" * 70)
    print(f"Processing: {sample_id}")
    print("=" * 70)

    # --------------------------------------------------------
    # Check matching files
    # --------------------------------------------------------

    if not height_path.exists():
        print(f"SKIPPED: AGL file not found")
        print(f"       {height_path}")
        return None

    if not class_path.exists():
        print(f"SKIPPED: CLS file not found")
        print(f"       {class_path}")
        return None

    # --------------------------------------------------------
    # Load depth
    # --------------------------------------------------------

    depth = np.load(depth_path).astype(np.float32)

    # --------------------------------------------------------
    # Load GAMUS ground truth
    # --------------------------------------------------------

    agl = load_h5_image(height_path).astype(np.float32)
    class_map = load_h5_image(class_path)

    class_map = np.rint(class_map).astype(np.int32)

    print(f"Depth shape:     {depth.shape}")
    print(f"AGL shape:       {agl.shape}")
    print(f"Class-map shape: {class_map.shape}")

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    if depth.shape != agl.shape:
        print("SKIPPED: depth and AGL shapes do not match")
        return None

    if depth.shape != class_map.shape:
        print("SKIPPED: depth and class-map shapes do not match")
        return None

    # --------------------------------------------------------
    # Valid pixels
    #
    # GAMUS:
    #   class 0 = background / others
    #
    # We exclude:
    #   - NaN / infinite depth
    #   - NaN / infinite AGL
    #   - negative AGL
    #   - class 0 background
    # --------------------------------------------------------

    valid = (
        np.isfinite(depth)
        & np.isfinite(agl)
        & (agl >= 0)
        & (class_map != 0)
    )

    valid_pixels = int(valid.sum())

    print(f"Valid pixels:    {valid_pixels:,}")

    if valid_pixels < 100:
        print("SKIPPED: too few valid pixels")
        return None

    # --------------------------------------------------------
    # Convert valid mask into the format expected by
    # terrain_aware_calibration()
    # --------------------------------------------------------

    void_mask = ~valid

    # --------------------------------------------------------
    # FINALIZED TERRAIN-AWARE POLYNOMIAL CALIBRATION
    #
    # This uses the existing function from
    # elevation_pipeline.py.
    #
    # Degree = 2
    # --------------------------------------------------------

    calibrated, fit_report = terrain_aware_calibration(
        depth=depth,
        reference_elevation=agl,
        void_mask=void_mask,
        terrain_mask=class_map,
        degree=POLYNOMIAL_DEGREE
    )

    # Do not treat background / invalid pixels as actual output
    calibrated = calibrated.astype(np.float32)
    calibrated[~valid] = np.nan

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    predicted_valid = calibrated[valid]
    actual_valid = agl[valid]

    error = predicted_valid - actual_valid

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))

    if len(predicted_valid) > 1:
        correlation = float(
            np.corrcoef(predicted_valid, actual_valid)[0, 1]
        )
    else:
        correlation = np.nan

    print(f"MAE:             {mae:.4f} m")
    print(f"RMSE:            {rmse:.4f} m")
    print(f"Correlation:     {correlation:.4f}")

    # --------------------------------------------------------
    # PER-CLASS FIT REPORT
    # --------------------------------------------------------

    print("\nPer-class polynomial fits:")

    for cls, (coeffs, n_pixels) in fit_report.items():

        cls_int = int(cls)
        class_name = CLASS_NAMES.get(cls_int, f"class_{cls_int}")

        coeffs_string = ", ".join(
            f"{float(c):.4f}" for c in coeffs
        )

        print(
            f"  {class_name}: "
            f"coeffs=[{coeffs_string}], "
            f"pixels={n_pixels:,}"
        )

    # --------------------------------------------------------
    # SAVE OUTPUT ELEVATION / HEIGHT ARRAY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_npy = OUTPUT_DIR / f"{sample_id}_polynomial_elevation.npy"

    np.save(output_npy, calibrated)

    # --------------------------------------------------------
    # SAVE PREVIEW IMAGE
    # --------------------------------------------------------

    output_png = OUTPUT_DIR / f"{sample_id}_polynomial_preview.png"

    plt.figure(figsize=(10, 8))

    image = plt.imshow(calibrated, cmap="terrain")

    plt.colorbar(
        image,
        label="Calibrated Height (m)"
    )

    plt.title(
        f"{sample_id}\n"
        f"Terrain-aware Polynomial Calibration (Degree 2)"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_png,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nSaved:")
    print(f"  {output_npy}")
    print(f"  {output_png}")

    return {
        "sample_id": sample_id,
        "valid_pixels": valid_pixels,
        "mae": mae,
        "rmse": rmse,
        "correlation": correlation,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GAMUS PRODUCTION TEST")
    print("Terrain-aware Polynomial Calibration")
    print("=" * 70)

    depth_files = sorted(
        DEPTH_DIR.glob("*_RGB_depth.npy")
    )

    print(f"\nDepth directory:")
    print(f"  {DEPTH_DIR}")

    print(f"\nDepth files found: {len(depth_files)}")

    if not depth_files:
        print("\nNo depth files found.")
        return

    results = []

    for depth_path in depth_files:

        result = process_sample(depth_path)

        if result is not None:
            results.append(result)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"Depth files found:       {len(depth_files)}")
    print(f"Samples successfully run: {len(results)}")
    print(f"Samples skipped:          {len(depth_files) - len(results)}")

    if not results:
        print("\nNo samples were successfully processed.")
        return

    print("\nPer-sample results:")
    print("-" * 70)

    for result in results:

        print(
            f"{result['sample_id']:<15} "
            f"MAE={result['mae']:>8.4f} m   "
            f"RMSE={result['rmse']:>8.4f} m   "
            f"Corr={result['correlation']:>7.4f}"
        )

    # --------------------------------------------------------
    # Aggregate statistics across samples
    # --------------------------------------------------------

    maes = np.array([r["mae"] for r in results])
    rmses = np.array([r["rmse"] for r in results])
    correlations = np.array(
        [r["correlation"] for r in results]
    )

    print("\n" + "-" * 70)
    print("Average across successfully processed samples:")
    print(f"Average MAE:         {np.mean(maes):.4f} m")
    print(f"Average RMSE:        {np.mean(rmses):.4f} m")
    print(f"Average correlation: {np.mean(correlations):.4f}")

    print("\nOutput directory:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()