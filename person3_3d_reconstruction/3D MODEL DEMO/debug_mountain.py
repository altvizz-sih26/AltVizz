"""
Standalone debug: runs the SAME processing pipeline as create_terrain.py
directly on your elevation .npy file, and prints mountain-region stats
at every stage. No images or GLB needed -- just run this on
elevation_after.npy to see exactly where the mountain's uplift is
getting lost.

Usage:
    python debug_mountain.py elevation_after.npy
"""

import sys
import numpy as np
from scipy.ndimage import (
    median_filter, gaussian_filter, binary_opening, binary_dilation,
    distance_transform_edt, label, find_objects,
)

NODATA_VALUE = -9999.0


def load_elevation(path):
    elevation = np.load(path).astype(np.float32)
    valid = np.isfinite(elevation) & (elevation != NODATA_VALUE)
    if not np.all(valid):
        _, indices = distance_transform_edt(~valid, return_indices=True)
        elevation = elevation[tuple(indices)]
    return elevation


def clip_elevation_outliers(elevation, low_pct=1.0, high_pct=99.0):
    lo = np.percentile(elevation, low_pct)
    hi = np.percentile(elevation, high_pct)
    return np.clip(elevation, lo, hi)


def _auto_base_kernel_size(height, width):
    size = max(9, min(height, width) // 35)
    if size % 2 == 0:
        size += 1
    return size


def separate_relief(elevation, spike_smooth_sigma=1.2, min_feature_px=None,
                     mad_multiplier=5.0, mountain_detail_boost=1.3,
                     large_feature_multiplier=3.0):
    H, W = elevation.shape
    base_kernel = _auto_base_kernel_size(H, W)
    if min_feature_px is None:
        min_feature_px = max(3, base_kernel // 5)

    denoised = median_filter(elevation, size=3, mode="nearest")
    base = median_filter(denoised, size=base_kernel, mode="nearest")
    ground_floor = np.percentile(base, 5)

    broad_silhouette = gaussian_filter(base, sigma=base_kernel)
    ridge_detail = base - broad_silhouette
    r_lo, r_hi = np.percentile(ridge_detail, [2, 98])
    ridge_detail = np.clip(ridge_detail, r_lo, r_hi)
    base = broad_silhouette + ridge_detail * mountain_detail_boost
    base = np.clip(base, ground_floor, None)

    detail = denoised - base

    local_stat_kernel = base_kernel * 3
    if local_stat_kernel % 2 == 0:
        local_stat_kernel += 1
    local_median = median_filter(detail, size=local_stat_kernel, mode="nearest")
    local_mad = median_filter(np.abs(detail - local_median), size=local_stat_kernel, mode="nearest")
    global_mad = np.median(np.abs(detail - np.median(detail)))
    global_floor = global_mad if global_mad > 1e-6 else np.std(detail) * 1.0
    noise_threshold = mad_multiplier * np.maximum(local_mad, global_floor * 0.5)
    kept_mask = np.abs(detail) > noise_threshold

    structure = np.ones((min_feature_px, min_feature_px))
    kept_mask = binary_opening(kept_mask, structure=structure)

    dilate_px = max(1, min_feature_px // 2)
    kept_mask = binary_dilation(kept_mask, structure=np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1)))

    detail = np.clip(detail, 0, None)
    kept_mask = kept_mask & (detail > 0)

    kept_values = detail[kept_mask]
    if kept_values.size > 0:
        height_cap = np.percentile(kept_values, 95)
        if height_cap > 0:
            detail = np.minimum(detail, height_cap)

    large_feature_min_extent = base_kernel * large_feature_multiplier
    cleaned_detail = np.zeros_like(detail)
    labeled, num_features = label(kept_mask)

    if num_features > 0:
        objects = find_objects(labeled)
        for comp_id, sl in enumerate(objects, start=1):
            if sl is None:
                continue
            r0, r1 = sl[0].start, sl[0].stop
            c0, c1 = sl[1].start, sl[1].stop
            comp_extent = max(r1 - r0, c1 - c0)
            if comp_extent >= large_feature_min_extent:
                sigma = max(spike_smooth_sigma, comp_extent / 8.0)
            else:
                sigma = spike_smooth_sigma
            pad = int(sigma * 4) + 2
            pr0, pr1 = max(0, r0 - pad), min(H, r1 + pad)
            pc0, pc1 = max(0, c0 - pad), min(W, c1 + pad)
            local_mask = labeled[pr0:pr1, pc0:pc1] == comp_id
            local_detail = np.where(local_mask, detail[pr0:pr1, pc0:pc1], 0.0)
            if sigma > 0:
                local_detail = gaussian_filter(local_detail, sigma=sigma)
            cleaned_detail[pr0:pr1, pc0:pc1] += local_detail

    cleaned_detail = np.clip(cleaned_detail, 0, None)
    return (base + cleaned_detail).astype(np.float32), num_features, base_kernel


def calculate_relief_scale(elevation, mesh_width, mesh_height, target_pct=0.20):
    p2 = np.percentile(elevation, 2)
    p98 = np.percentile(elevation, 98)
    relief = float(p98 - p2)
    if relief <= 0:
        return 1.0
    horizontal_size = max(mesh_width, mesh_height)
    target_relief = horizontal_size * target_pct
    scale = target_relief / relief
    scale = max(scale, 1.5)
    scale = min(scale, 18.0)
    return float(scale)


def block_mean_downsample(elevation, step):
    if step <= 1:
        return elevation
    H, W = elevation.shape
    newH = H - (H % step)
    newW = W - (W % step)
    cropped = elevation[:newH, :newW]
    reshaped = cropped.reshape(newH // step, step, newW // step, step)
    return reshaped.mean(axis=(1, 3)).astype(np.float32)


def region_stats(arr, label_str):
    relief = np.percentile(arr, 98) - np.percentile(arr, 2)
    print(f"[{label_str}] min={arr.min():.3f} max={arr.max():.3f} "
          f"relief(p2-p98)={relief:.3f}")
    return relief


def debug(path, step=2, border_trim_pct=0.05, clip_low=1.0, clip_high=99.0):
    print(f"Loading {path} ...")
    elevation = load_elevation(path)
    H, W = elevation.shape
    print(f"Shape: {elevation.shape}\n")

    # mountain region assumption: right side, roughly rows 30%-90%,
    # cols 55%-100%. Adjust these fractions if your mountain sits
    # elsewhere in the image.
    r0, r1 = int(H * 0.3), int(H * 0.9)
    c0, c1 = int(W * 0.55), W
    print(f"Using mountain region: rows {r0}:{r1}, cols {c0}:{c1}\n")

    print("--- STAGE 1: raw loaded elevation ---")
    region_stats(elevation[r0:r1, c0:c1], "raw")

    # border trim (percentage-based, applied to whole grid)
    trim_h = int(round(H * border_trim_pct))
    trim_w = int(round(W * border_trim_pct))
    trimmed = elevation[trim_h:H - trim_h, trim_w:W - trim_w]
    Ht, Wt = trimmed.shape
    r0t, r1t = int(Ht * 0.3), int(Ht * 0.9)
    c0t, c1t = int(Wt * 0.55), Wt

    print("\n--- STAGE 2: after border trim ---")
    region_stats(trimmed[r0t:r1t, c0t:c1t], "after_trim")

    clipped = clip_elevation_outliers(trimmed, clip_low, clip_high)
    print("\n--- STAGE 3: after percentile clip (outlier removal) ---")
    region_stats(clipped[r0t:r1t, c0t:c1t], "after_clip")

    processed, num_features, base_kernel = separate_relief(clipped)
    print(f"\n--- STAGE 4: after separate_relief (base_kernel={base_kernel}, "
          f"{num_features} features detected in WHOLE image) ---")
    region_stats(processed[r0t:r1t, c0t:c1t], "after_separate_relief")

    downsampled = block_mean_downsample(processed, step)
    Hd, Wd = downsampled.shape
    r0d, r1d = int(Hd * 0.3), int(Hd * 0.9)
    c0d, c1d = int(Wd * 0.55), Wd
    print(f"\n--- STAGE 5: after downsample (step={step}) ---")
    relief_downsampled = region_stats(downsampled[r0d:r1d, c0d:c1d], "after_downsample")

    scale = calculate_relief_scale(downsampled, Wd, Hd, target_pct=0.20)
    print(f"\n--- STAGE 6: vertical scale ---")
    print(f"common_scale = {scale:.3f}")
    print(f"Final mountain relief AFTER vertical scale = {relief_downsampled * scale:.3f}")
    print(f"(mesh width/height = {Wd} x {Hd}, so this is how tall the "
          f"mountain will look relative to the mesh's horizontal size)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_mountain.py elevation_after.npy [step]")
        sys.exit(1)
    step_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    debug(sys.argv[1], step=step_arg)