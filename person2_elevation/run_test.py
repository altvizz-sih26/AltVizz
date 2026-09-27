import numpy as np
import matplotlib.pyplot as plt
from depth_loader import load_and_validate_depth
from elevation_pipeline import generate_elevation, CLASS_NAMES

# ---- ONLY place filenames should ever appear ----
DEPTH_PATH = "person2_elevation/vegetation_1_depth.npy"
SRTM_PATH = "person2_elevation/srtm_vegetation_aligned.tif"
TERRAIN_MASK_PATH = "person2_elevation/terrain_mask.npy"
OUTPUT_NAME = "vegetation"
# --------------------------------------------------

depth = load_and_validate_depth(DEPTH_PATH)
terrain_mask = np.load(TERRAIN_MASK_PATH)

elevation, meta = generate_elevation(depth, srtm_path=SRTM_PATH, terrain_mask=terrain_mask)

print(f"Mode used: {meta['mode']}")
print(f"Edge artifact pixels excluded: {meta['artifact_pixels']}")
if meta["mode"] != "relative":
    print(f"Mean absolute error: {meta['error_mean']:.2f} meters")
    print(f"Max absolute error: {meta['error_max']:.2f} meters")
    print(f"Correlation: {meta['correlation']:.3f}")
    print("Per-class fit report:")
    for cls, (a, b, n) in meta["fit_report"].items():
        name = CLASS_NAMES.get(cls, cls) if isinstance(cls, int) else cls
        print(f"  {name}: a={a:.2f}, b={b:.2f}, pixels={n}")

np.save(f"person2_elevation/final_elevation_{OUTPUT_NAME}.npy", elevation)

plt.figure(figsize=(6, 6))
plt.imshow(elevation, cmap='terrain')
plt.colorbar(label='Elevation (meters)')
title = f"{OUTPUT_NAME} ({meta['mode']})"
if meta["error_mean"] is not None:
    title += f" — Err: {meta['error_mean']:.1f}m, Corr: {meta['correlation']:.2f}"
plt.title(title)
plt.savefig(f"person2_elevation/final_elevation_{OUTPUT_NAME}_preview.png", dpi=150)

print(f"\nSaved: final_elevation_{OUTPUT_NAME}.npy + preview.png")