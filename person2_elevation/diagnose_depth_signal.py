import numpy as np
from pathlib import Path
from load_gamus import load_gamus_sample, discover_complete_samples

DEPTH_DIR = Path("person2_elevation/GAMUS_depth")
CLASS_NAMES = {1: "ground", 2: "low_vegetation", 3: "building", 4: "water", 5: "road", 6: "tree"}
CHECK_CLASSES = [3, 6]  # building, tree — largest classes, most informative

def load_depth(sample_id):
    return np.load(DEPTH_DIR / f"{sample_id}_RGB_depth.npy")

def diagnose():
    sample_ids = discover_complete_samples()
    if not sample_ids:
        print("No samples found.")
        return

    for class_id in CHECK_CLASSES:
        name = CLASS_NAMES[class_id]
        per_sample_std, per_sample_range = [], []

        for sample_id in sample_ids:
            _, agl, class_map = load_gamus_sample(sample_id)
            depth = load_depth(sample_id)
            mask = class_map == class_id
            if mask.sum() < 50:
                continue
            d = depth[mask]
            per_sample_std.append(d.std())
            per_sample_range.append(d.max() - d.min())

        print(f"=== {name} (class {class_id}) ===")
        print(f"  samples checked: {len(per_sample_std)}")
        print(f"  avg within-image depth std:   {np.mean(per_sample_std):.4f}")
        print(f"  avg within-image depth range: {np.mean(per_sample_range):.4f}")
        print(f"  (depth is normalized 0-1 per image; near-0 std means the "
              f"model gives almost every {name} pixel in a photo nearly "
              f"the SAME depth value — i.e. it isn't telling individual "
              f"{name}s apart)")
        print()

if __name__ == "__main__":
    diagnose()