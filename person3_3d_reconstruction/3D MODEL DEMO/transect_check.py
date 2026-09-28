"""
Prints a horizontal transect of REAL elevation values straight through
the mountain area, so we can see the actual raw shape (smooth slope vs
blocky/terraced) before any processing touches it.

Usage:
    python transect_check.py elevation_after.npy
"""

import sys
import numpy as np

def transect(path):
    elevation = np.load(path).astype(np.float32)
    H, W = elevation.shape
    print(f"Shape: {elevation.shape}")

    # Right-side region (adjust these fractions if your mountain sits
    # elsewhere in the image)
    row = int(H * 0.6)          # a row roughly 60% down the image
    col_start = int(W * 0.55)   # starting from the right half
    col_end = W

    print(f"\nTransect along row={row}, columns {col_start} to {col_end} "
          f"(every 10 pixels):")
    line = elevation[row, col_start:col_end:10]
    print(np.round(line, 2))

    # Step sizes between consecutive sampled points -- a smooth slope
    # has small, gradually changing steps; blocky/terraced data has a
    # few very large jumps between otherwise-flat runs.
    steps = np.diff(line)
    print("\nStep sizes between samples:")
    print(np.round(steps, 2))
    print(f"\nMax single step: {np.max(np.abs(steps)):.2f}")
    print(f"Mean |step|: {np.mean(np.abs(steps)):.2f}")

    # Also check vertical transect through the same region
    col = int(W * 0.75)
    row_start = int(H * 0.3)
    row_end = int(H * 0.9)
    print(f"\nVertical transect along col={col}, rows {row_start} to "
          f"{row_end} (every 10 pixels):")
    vline = elevation[row_start:row_end:10, col]
    print(np.round(vline, 2))
    vsteps = np.diff(vline)
    print(f"Max single vertical step: {np.max(np.abs(vsteps)):.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transect_check.py elevation_after.npy")
        sys.exit(1)
    transect(sys.argv[1])