import h5py
import numpy as np
from pathlib import Path

HEIGHTS_DIR = Path("gamus_test/heights")
CLASSES_DIR = Path("gamus_test/classes")

agl_files = sorted(HEIGHTS_DIR.glob("*_AGL.h5"))
print(f"Scanning {len(agl_files)} AGL files...\n")

all_agl_min, all_agl_max = [], []
neg_pixel_total, total_pixel_total = 0, 0
class_counts = np.zeros(8, dtype=np.int64)

for agl_path in agl_files:
    sample_id = agl_path.stem.replace("_AGL", "")
    cls_path = CLASSES_DIR / f"{sample_id}_CLS.h5"

    with h5py.File(agl_path, "r") as f:
        agl = f["image"][()]
    with h5py.File(cls_path, "r") as f:
        cls = f["image"][()].astype(int)

    all_agl_min.append(agl.min())
    all_agl_max.append(agl.max())
    neg_pixel_total += (agl < 0).sum()
    total_pixel_total += agl.size

    for c in np.unique(cls):
        if 0 <= c < len(class_counts):
            class_counts[c] += (cls == c).sum()

print("=== AGL value range across all 50 samples ===")
print(f"Overall min: {min(all_agl_min):.3f}")
print(f"Overall max: {max(all_agl_max):.3f}")
print(f"Negative AGL pixels: {neg_pixel_total} / {total_pixel_total} "
      f"({100*neg_pixel_total/total_pixel_total:.3f}%)")

print("\n=== Class ID pixel counts across all 50 samples ===")
for c, count in enumerate(class_counts):
    if count > 0:
        pct = 100 * count / total_pixel_total
        print(f"Class {c}: {count:>10} pixels ({pct:5.2f}%)")