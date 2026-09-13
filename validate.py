import rasterio
import numpy as np
import csv

def compute_metrics(predicted_path, reference_path):
    with rasterio.open(predicted_path) as pred_src:
        predicted = pred_src.read(1).astype(np.float32)
    with rasterio.open(reference_path) as ref_src:
        reference = ref_src.read(1).astype(np.float32)

    if predicted.shape != reference.shape:
        raise ValueError(f"Shape mismatch: predicted {predicted.shape} vs reference {reference.shape}")

    diff = predicted - reference
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    corr = np.corrcoef(predicted.flatten(), reference.flatten())[0, 1]
    return mae, rmse, corr

# --- CONFIG ---
# Update predicted_path for each region once real outputs land.
regions = {
    "urban": {
        "predicted_path": "test_data/urban/srtm_urban_aligned.tif",   # <-- replace with real predicted
        "reference_path": "test_data/urban/srtm_urban_aligned.tif",
    },
    "vegetation": {
        "predicted_path": "test_data/vegetation/srtm_vegetation_aligned.tif",
        "reference_path": "test_data/vegetation/srtm_vegetation_aligned.tif",
    },
    "bare_terrain": {
        "predicted_path": "test_data/bare_terrain/srtm_bare_terrain_aligned.tif",
        "reference_path": "test_data/bare_terrain/srtm_bare_terrain_aligned.tif",
    },
}

results = []
for name, paths in regions.items():
    mae, rmse, corr = compute_metrics(paths["predicted_path"], paths["reference_path"])
    results.append({"region": name, "mae": mae, "rmse": rmse, "correlation": corr})

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