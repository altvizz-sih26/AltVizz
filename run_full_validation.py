"""
run_full_validation.py

Batch validation across EVERY image in the dataset (not a handful) - the
direct answer to "you only tested 3-4 images."

For each image, independently computes MAE, RMSE, and Correlation against
SRTM reference elevation - not trusting only the calibration step's
self-reported numbers. Terrain type is inferred from the folder each image
lives in (test_data/urban/, test_data/vegetation/, test_data/bare_terrain/,
or any new folder you add - the folder name becomes the terrain label).

Usage:
    python run_full_validation.py --data-root test_data --predicted-dir person3_3d_reconstruction
"""

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import rasterio

SIH_ROOT = os.path.abspath(os.path.dirname(__file__))
if SIH_ROOT not in sys.path:
    sys.path.insert(0, SIH_ROOT)

from process_new_image import process_new_image

NODATA_VALUE = -9999.0


def compute_metrics(predicted, reference):
    if predicted.shape != reference.shape:
        raise ValueError("Shape mismatch: predicted " + str(predicted.shape) + " vs reference " + str(reference.shape))

    valid_mask = (reference != NODATA_VALUE) & np.isfinite(reference) & np.isfinite(predicted)
    predicted_v = predicted[valid_mask]
    reference_v = reference[valid_mask]

    if predicted_v.size < 2:
        raise ValueError("Not enough valid overlapping pixels to compute metrics")

    diff = predicted_v - reference_v
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    corr = float(np.corrcoef(predicted_v, reference_v)[0, 1])
    return mae, rmse, corr, int(valid_mask.sum()), int(reference.size)


def load_reference(image_path, srtm_cache_dir):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cached_path = os.path.join(srtm_cache_dir, "srtm_" + base_name + "_aligned.tif")
    if os.path.isfile(cached_path):
        return cached_path
    os.makedirs(srtm_cache_dir, exist_ok=True)
    result_path = process_new_image(image_path)
    if os.path.abspath(result_path) != os.path.abspath(cached_path):
        import shutil
        shutil.move(result_path, cached_path)
        raw_path = "srtm_" + base_name + "_raw.tif"
        if os.path.isfile(raw_path):
            os.remove(raw_path)
    return cached_path


def find_predicted_elevation(image_path, predicted_dir, terrain_type=None):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    candidates = []
    if predicted_dir:
        candidates.append(os.path.join(predicted_dir, "final_elevation_" + base_name + ".npy"))
        candidates.append(os.path.join(predicted_dir, base_name + "_elevation.npy"))
        candidates.append(os.path.join(predicted_dir, base_name + ".npy"))
        if terrain_type:
            candidates.append(os.path.join(predicted_dir, "final_elevation_" + terrain_type + ".npy"))
            candidates.append(os.path.join(predicted_dir, terrain_type + "_elevation.npy"))
            candidates.append(os.path.join(predicted_dir, terrain_type + ".npy"))
    candidates.append(os.path.join(SIH_ROOT, "app", "results", base_name + "_elevation.npy"))

    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def run_live_pipeline(image_path, srtm_path):
    from app.services.depth_pipeline import run_pipeline
    result = run_pipeline(image_path, srtm_path=srtm_path)
    elevation_path = result.get("elevation_array_path")
    if elevation_path is None or not os.path.isfile(elevation_path):
        raise RuntimeError("Live pipeline did not produce a saved elevation array")
    return elevation_path


def discover_images(data_root):
    images = []
    for terrain_folder in sorted(os.listdir(data_root)):
        folder_path = os.path.join(data_root, terrain_folder)
        if not os.path.isdir(folder_path):
            continue
        for ext in ("*.tif", "*.tiff"):
            for img_path in sorted(glob.glob(os.path.join(folder_path, ext))):
                fname = os.path.basename(img_path).lower()
                if fname.startswith("srtm_"):
                    continue
                images.append({"path": img_path, "terrain_type": terrain_folder})
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="test_data")
    parser.add_argument("--predicted-dir", default=None)
    parser.add_argument("--srtm-cache-dir", default="srtm_cache")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--out", default="full_validation_report.csv")
    args = parser.parse_args()

    images = discover_images(args.data_root)
    if not images:
        print("No images found under " + args.data_root + ". Check the folder structure.")
        sys.exit(1)

    terrain_folder_count = len(set(item["terrain_type"] for item in images))
    print("Found " + str(len(images)) + " images across " + str(terrain_folder_count) + " terrain folders.")
    print("")

    rows = []
    for i, item in enumerate(images, 1):
        image_path = item["path"]
        terrain_type = item["terrain_type"]
        name = os.path.basename(image_path)
        print("[" + str(i) + "/" + str(len(images)) + "] " + terrain_type + "/" + name + " ...", end=" ", flush=True)

        row = {"image": name, "terrain_type": terrain_type, "status": "ok",
               "mae": None, "rmse": None, "correlation": None,
               "valid_pixels": None, "total_pixels": None, "seconds": None}
        t0 = time.time()
        try:
            srtm_path = load_reference(image_path, args.srtm_cache_dir)

            elevation_path = find_predicted_elevation(image_path, args.predicted_dir, terrain_type)
            if elevation_path is None:
                if not args.live:
                    row["status"] = "skipped_no_predicted_elevation"
                    print("SKIPPED (no precomputed elevation, use --live to generate)")
                    rows.append(row)
                    continue
                elevation_path = run_live_pipeline(image_path, srtm_path)

            predicted = np.load(elevation_path).astype(np.float32)
            with rasterio.open(srtm_path) as src:
                reference = src.read(1).astype(np.float32)

            mae, rmse, corr, n_valid, n_total = compute_metrics(predicted, reference)
            row.update({"mae": mae, "rmse": rmse, "correlation": corr,
                        "valid_pixels": n_valid, "total_pixels": n_total})
            print("MAE=" + format(mae, ".2f") + "m RMSE=" + format(rmse, ".2f") + "m Corr=" + format(corr, ".3f"))

        except Exception as e:
            row["status"] = "error: " + str(e)
            print("FAILED (" + str(e) + ")")

        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("")
    print("Saved per-image report: " + args.out)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    print("")
    print("=" * 70)
    print("SUMMARY  (" + str(len(ok_rows)) + "/" + str(len(rows)) + " images scored successfully)")
    print("=" * 70)
    print(f"{'Terrain':<15}{'N':<5}{'Avg MAE':<12}{'Avg RMSE':<12}{'Avg Corr':<12}")
    print("-" * 56)

    terrain_types = sorted(set(r["terrain_type"] for r in ok_rows))
    summary_rows = []
    for t in terrain_types:
        group = [r for r in ok_rows if r["terrain_type"] == t]
        avg_mae = np.mean([r["mae"] for r in group])
        avg_rmse = np.mean([r["rmse"] for r in group])
        avg_corr = np.mean([r["correlation"] for r in group])
        print(f"{t:<15}{len(group):<5}{avg_mae:<12.3f}{avg_rmse:<12.3f}{avg_corr:<12.3f}")
        summary_rows.append({"terrain_type": t, "n_images": len(group),
                              "avg_mae": avg_mae, "avg_rmse": avg_rmse, "avg_correlation": avg_corr})

    if ok_rows:
        overall_mae = np.mean([r["mae"] for r in ok_rows])
        overall_rmse = np.mean([r["rmse"] for r in ok_rows])
        overall_corr = np.mean([r["correlation"] for r in ok_rows])
        print("-" * 56)
        print(f"{'OVERALL':<15}{len(ok_rows):<5}{overall_mae:<12.3f}{overall_rmse:<12.3f}{overall_corr:<12.3f}")
        summary_rows.append({"terrain_type": "OVERALL", "n_images": len(ok_rows),
                              "avg_mae": overall_mae, "avg_rmse": overall_rmse, "avg_correlation": overall_corr})

    skipped = [r for r in rows if r["status"] != "ok"]
    if skipped:
        print("")
        print(str(len(skipped)) + " image(s) not scored:")
        for r in skipped:
            print("  - " + r["terrain_type"] + "/" + r["image"] + ": " + r["status"])

    summary_out = args.out.replace(".csv", "_summary.csv")
    with open(summary_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["terrain_type", "n_images", "avg_mae", "avg_rmse", "avg_correlation"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print("Saved summary report: " + summary_out)


if __name__ == "__main__":
    main()
