import h5py
import numpy as np
from pathlib import Path
from PIL import Image


GAMUS_DIR = Path("gamus_test")
DEFAULT_DEPTH_DIR = Path("person2_elevation/GAMUS_depth")


def discover_complete_samples(depth_dir=DEFAULT_DEPTH_DIR):
    """
    Finds every GAMUS sample for which all required files are present:

        Person 1 depth:
            <sample_id>_RGB_depth.npy

        GAMUS RGB:
            <sample_id>.png

        GAMUS AGL:
            <sample_id>_AGL.h5

        GAMUS classes:
            <sample_id>_CLS.h5

    Returns a sorted list of sample IDs that have all four files.
    """

    depth_dir = Path(depth_dir)
    depth_files = sorted(depth_dir.glob("*_RGB_depth.npy"))

    if not depth_files:
        print(f"No depth files found in {depth_dir} yet — nothing to check.")
        return []

    complete = []
    incomplete = []

    for depth_path in depth_files:

        # Example:
        # DC_03_26_RGB_depth.npy
        #       ↓
        # DC_03_26
        sample_id = depth_path.stem.replace("_RGB_depth", "")

        rgb_path = GAMUS_DIR / "rgb_png" / f"{sample_id}.png"
        agl_path = GAMUS_DIR / "heights" / f"{sample_id}_AGL.h5"
        cls_path = GAMUS_DIR / "classes" / f"{sample_id}_CLS.h5"

        missing = []

        if not rgb_path.exists():
            missing.append("RGB")

        if not agl_path.exists():
            missing.append("AGL")

        if not cls_path.exists():
            missing.append("CLS")

        if not missing:
            complete.append(sample_id)
        else:
            incomplete.append((sample_id, missing))

    print(f"Depth files found: {len(depth_files)}")
    print(f"Complete samples (all 4 files present): {len(complete)}")

    if incomplete:
        print(f"Incomplete samples (skipped): {len(incomplete)}")

        for sample_id, missing in incomplete:
            print(f"  {sample_id}: missing {missing}")

    return sorted(complete)


def load_gamus_sample(sample_id):
    """
    Loads the GAMUS RGB image, AGL ground truth, and GAMUS
    ground-truth class map for one sample.

    Returns:
        rgb        : H x W x 3 uint8 array
        agl        : H x W float32 array
        class_map  : H x W class-label array
    """

    rgb_path = GAMUS_DIR / "rgb_png" / f"{sample_id}.png"
    agl_path = GAMUS_DIR / "heights" / f"{sample_id}_AGL.h5"
    cls_path = GAMUS_DIR / "classes" / f"{sample_id}_CLS.h5"

    if not rgb_path.exists():
        raise FileNotFoundError(
            f"RGB file not found: {rgb_path}"
        )

    if not agl_path.exists():
        raise FileNotFoundError(
            f"AGL file not found: {agl_path}"
        )

    if not cls_path.exists():
        raise FileNotFoundError(
            f"Class file not found: {cls_path}"
        )

    # Load RGB PNG
    rgb = np.array(Image.open(rgb_path))

    # Load GAMUS AGL
    with h5py.File(agl_path, "r") as f:
        agl = f["image"][()]

    # Load GAMUS class map
    with h5py.File(cls_path, "r") as f:
        class_map = f["image"][()]

    # Check spatial dimensions
    if rgb.shape[:2] != agl.shape:
        raise ValueError(
            f"RGB and AGL dimensions do not match: "
            f"{rgb.shape} vs {agl.shape}"
        )

    if agl.shape != class_map.shape:
        raise ValueError(
            f"AGL and class-map dimensions do not match: "
            f"{agl.shape} vs {class_map.shape}"
        )

    return rgb, agl, class_map


def get_valid_mask(agl, class_map):
    """
    Returns a boolean mask of pixels usable for calibration.

    Excludes:

    1. Negative AGL values.
       Based on the GAMUS inspection already performed, negative
       values are treated as invalid for this calibration experiment.

    2. GAMUS class 0.
       Class 0 represents background / non-target pixels and is
       excluded from terrain calibration.

    Valid GAMUS terrain classes are therefore:
        1 = Ground
        2 = Low vegetation
        3 = Buildings
        4 = Water
        5 = Road
        6 = Tree
    """

    valid_agl = np.isfinite(agl) & (agl >= 0)

    valid_class = (
        np.isfinite(class_map)
        & (class_map.astype(int) != 0)
    )

    return valid_agl & valid_class


if __name__ == "__main__":

    sample_id = "DC_03_26"

    print("Checking GAMUS sample:", sample_id)

    rgb, agl, class_map = load_gamus_sample(sample_id)

    print("\nRGB")
    print("  shape:", rgb.shape)
    print("  dtype:", rgb.dtype)
    print("  min:", rgb.min())
    print("  max:", rgb.max())

    print("\nAGL")
    print("  shape:", agl.shape)
    print("  dtype:", agl.dtype)
    print("  min:", agl.min())
    print("  max:", agl.max())

    print("\nClass map")
    print("  shape:", class_map.shape)
    print("  dtype:", class_map.dtype)
    print("  classes:", np.unique(class_map))

    mask = get_valid_mask(agl, class_map)

    print("\nValid pixel mask")
    print(
        "  valid pixels:",
        mask.sum(),
        "/",
        mask.size,
        f"({100 * mask.sum() / mask.size:.2f}%)"
    )

    print("\nDepth discovery test")

    sample_ids = discover_complete_samples()

    if sample_ids:
        print("\nFirst available samples:")

        for sample_id in sample_ids[:10]:
            print(f"  {sample_id}")

        if len(sample_ids) > 10:
            print(f"  ... and {len(sample_ids) - 10} more")

    else:
        print("No complete samples found.")