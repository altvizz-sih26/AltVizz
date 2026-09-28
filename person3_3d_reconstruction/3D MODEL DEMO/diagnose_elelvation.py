"""
Quick diagnostic: run this on your DURING elevation .npy file to find
out WHY the mountain area is coming out as a flat plateau + spikes.

Usage:
    python diagnose_elevation.py path/to/during_elevation.npy
"""

import sys
import numpy as np

def diagnose(path):
    elevation = np.load(path)
    elevation = np.asarray(elevation, dtype=np.float32)

    print(f"Shape: {elevation.shape}")
    print(f"dtype (original file): checking raw load...")

    finite = np.isfinite(elevation)
    print(f"NaN/Inf count: {np.sum(~finite)}")

    print(f"\nMin: {elevation.min()}")
    print(f"Max: {elevation.max()}")
    print(f"Mean: {elevation.mean():.3f}")
    print(f"Median: {np.median(elevation):.3f}")

    # Check for common NODATA sentinels
    candidates = [-9999.0, -9999, -32768, -32767, 0.0, -3.4028235e+38]
    print("\n--- Checking common NODATA sentinel candidates ---")
    for c in candidates:
        count = np.sum(np.isclose(elevation, c, atol=1e-3))
        pct = 100.0 * count / elevation.size
        if count > 0:
            print(f"  value ~= {c}: {count} pixels ({pct:.2f}%)")

    # Histogram of unique-ish values to spot "flat plateau" (one dominant value)
    flat_vals, counts = np.unique(np.round(elevation, 1), return_counts=True)
    top_idx = np.argsort(counts)[::-1][:10]
    print("\n--- Top 10 most common elevation values (rounded to 0.1) ---")
    for i in top_idx:
        pct = 100.0 * counts[i] / elevation.size
        print(f"  value={flat_vals[i]:.1f}  count={counts[i]}  ({pct:.2f}% of image)")

    # Percentile spread (relief) actually present in the data
    p2, p50, p98 = np.percentile(elevation, [2, 50, 98])
    print(f"\nPercentile spread: p2={p2:.3f}  p50={p50:.3f}  p98={p98:.3f}")
    print(f"p98 - p2 relief = {p98 - p2:.3f}")

    # If one single value covers a huge % of the image, that's your flat-fill culprit
    dominant_pct = 100.0 * counts[top_idx[0]] / elevation.size
    if dominant_pct > 30:
        print(f"\n[WARNING] A single elevation value covers {dominant_pct:.1f}% of the "
              f"image -- this strongly suggests a flat-fill (NODATA sentinel not being "
              f"caught, or a huge NODATA region) rather than real varied mountain terrain.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python diagnose_elevation.py path/to/elevation.npy")
        sys.exit(1)
    diagnose(sys.argv[1])