"""
Honest calibration validator for a labeled image that has real SRTM.

Usage (same naming as the runner):
    python validate_new_dataset.py --name santa-rosa-wildfire_00000000_pre_disaster \
        --srtm srtm_santa-rosa-wildfire_00000000_aligned.tif

Files read (from --dir, default "."):
    <name>_height_normalized.npy   Person 1, values 0-1
    <name>_terrain_mask.npy        Person 1
    --srtm                         real SRTM tif, same shape as the depth

Never point this at a _height_meters.npy file: those are metres, not 0-1.
Does not touch elevation_pipeline.py or production behaviour.
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio

from fit_methods import fit_terrain_wise, predict_terrain_wise
from metrics import compute_metrics

# Use the classifier's own class ids if elevation_pipeline exposes them.
# Otherwise fall back to generic labels built from the ids found in the mask,
# so no guessed names can mislabel a class.
try:
    from elevation_pipeline import CLASS_NAMES
except ImportError:
    CLASS_NAMES = None


def load_reference_elevation(srtm_path, shape):
    with rasterio.open(srtm_path) as ds:
        ref = ds.read(1).astype(np.float32)
    if ref.shape != shape:
        raise ValueError(
            f"SRTM shape {ref.shape} doesn't match depth shape {shape}. "
            "Align/resample the SRTM first."
        )
    return ref


def block_test_mask(shape, block, test_fraction, seed):
    """Hold out whole blocks, so neighbouring (correlated) pixels can't leak."""
    rng = np.random.default_rng(seed)
    nby, nbx = -(-shape[0] // block), -(-shape[1] // block)
    block_is_test = rng.random((nby, nbx)) < test_fraction
    up = np.kron(block_is_test, np.ones((block, block), dtype=bool))
    return up[:shape[0], :shape[1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="image name prefix, as used by the runner")
    ap.add_argument("--srtm", required=True, help="path to the aligned SRTM .tif")
    ap.add_argument("--dir", default=".", help="folder holding Person 1's .npy files")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--block", type=int, default=64, help="holdout block size in pixels")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = Path(args.dir)
    depth_path = d / f"{args.name}_height_normalized.npy"
    mask_path = d / f"{args.name}_terrain_mask.npy"
    for p in (depth_path, mask_path, Path(args.srtm)):
        if not p.exists():
            raise FileNotFoundError(p)

    depth = np.load(depth_path).astype(np.float32)
    terrain_mask = np.load(mask_path)
    if depth.min() < 0 or depth.max() > 1.0001:
        raise ValueError(
            f"{depth_path.name} has range {depth.min():.2f}..{depth.max():.2f}, "
            "not 0-1. This looks like a height_meters file; use height_normalized."
        )
    if terrain_mask.shape != depth.shape:
        raise ValueError(f"mask shape {terrain_mask.shape} != depth shape {depth.shape}")

    ref = load_reference_elevation(args.srtm, depth.shape)

    valid = np.isfinite(depth) & np.isfinite(ref)
    print(f"[{args.name}] valid pixels: {valid.sum()} / {valid.size}")

    # Honest split: decided BEFORE any fitting, by whole blocks
    is_test = valid & block_test_mask(depth.shape, args.block, args.test_fraction, args.seed)
    is_train = valid & ~is_test
    print(f"[{args.name}] train pixels: {is_train.sum()}, test pixels: {is_test.sum()}")

    fit_report = fit_terrain_wise(depth[is_train], ref[is_train], terrain_mask[is_train],
                                  model_type="huber", degree=2)
    predicted_test = predict_terrain_wise(depth[is_test], terrain_mask[is_test], fit_report)
    ref_test, class_test = ref[is_test], terrain_mask[is_test]

    print(f"\n=== HONEST held-out results for {args.name} (never seen during fitting) ===")
    pooled = compute_metrics(predicted_test, ref_test)
    print(f"Pooled: corr={pooled['correlation']:.3f}  mae={pooled['mae']:.3f}  "
          f"rmse={pooled['rmse']:.3f}  n={pooled['n']}")

    print("\nPer terrain class:")
    for class_id in np.unique(class_test):
        name = (CLASS_NAMES or {}).get(int(class_id), f"class_{int(class_id)}")
        sel = class_test == class_id
        if sel.sum() < 10:
            print(f"  {name:16s}: too few test pixels ({sel.sum()}), skipped")
            continue
        m = compute_metrics(predicted_test[sel], ref_test[sel])
        print(f"  {name:16s}: corr={m['correlation']:.3f}  mae={m['mae']:.3f}  "
              f"rmse={m['rmse']:.3f}  n={m['n']}")

    print("\nThis is the accuracy number to report: it was never shown to the fit.")


if __name__ == "__main__":
    main()