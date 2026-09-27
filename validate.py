import rasterio
import numpy as np
import csv

NODATA_VALUE = -9999.0

def load_predicted(path):
    """Person 2's outputs are .npy files (raw elevation arrays)."""
    return np.load(path).astype(np.float32)

def load_reference(path):
    """SRTM aligned files are .tif, may contain NODATA_VALUE for edge pixels."""
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)

def compute_metrics(predicted_path, reference_path):
    predicted = load_predicted(predicted_path)
    reference = load_reference(reference_path)

    if predicted.shape != reference.shape:
        raise ValueError(f"Shape mismatch: predicted {predicted.shape} vs reference {reference.shape}")

    valid_mask = (reference != NODATA_VALUE) & np.isfinite(reference) & np.isfinite(predicted)
    predicted_v = predicted[valid_mask]
    reference_v = reference[valid_mask]

    diff = predicted_v - reference_v
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    corr = np.corrcoef(predicted_v, reference_v)[0, 1]
    n_valid = valid_mask.sum()
    n_total = reference.size
    return mae, rmse, corr, n_valid, n_total

# --- CONFIG ---
# predicted_path = Person 2's calibrated elevation .npy file for that region
# reference_path = our aligned SRTM .tif for that region
regions = {
    "urban": {
        "predicted_path": "final_elevation_urban.npy",
        "reference_path": "test_data/urban/srtm_urban_aligned.tif",
    },
    "vegetation": {
        "predicted_path": "final_elevation_vegetation.npy",
        "reference_path": "test_data/vegetation/srtm_vegetation_aligned.tif",
    },
    # "bare_terrain": {
    #     "predicted_path": "final_elevation_bare_terrain.npy",   # <-- add once Person 2 delivers this
    #     "reference_path": "test_data/bare_terrain/srtm_bare_terrain_aligned.tif",
    # },
}

results = []
for name, paths in regions.items():
    mae, rmse, corr, n_valid, n_total = compute_metrics(paths["predicted_path"], paths["reference_path"])
    results.append({"region": name, "mae": mae, "rmse": rmse, "correlation": corr})
    print(f"{name}: {n_valid}/{n_total} valid pixels used ({100*n_valid/n_total:.1f}%)")

print(f"\n{'Region':<15}{'MAE (m)':<12}{'RMSE (m)':<12}{'Correlation':<12}")
print("-" * 51)
for r in results:
    print(f"{r['region']:<15}{r['mae']:<12.3f}{r['rmse']:<12.3f}{r['correlation']:<12.3f}")

overall_mae = np.mean([r["mae"] for r in results])
overall_rmse = np.mean([r["rmse"] for r in results])
overall_corr = np.mean([r["correlation"] for r in results])
print("-" * 51)
print(f"{'OVERALL':<15}{overall_mae:<12.3f}{overall_rmse:<12.3f}{overall_corr:<12.3f}")

with open("validation_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["region", "mae", "rmse", "correlation"])
    writer.writeheader()
    writer.writerows(results)
    writer.writerow({"region": "OVERALL", "mae": overall_mae, "rmse": overall_rmse, "correlation": overall_corr})

print("\nSaved: validation_results.csv")