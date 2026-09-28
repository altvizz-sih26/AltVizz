from elevation_pipeline import generate_elevation, CLASS_NAMES
from confidence import compute_confidence
from diff_grid import compute_difference_grid
from pathlib import Path
import numpy as np


def _resolve_mask(depth_shape, terrain_mask):
    """If no terrain mask is available yet, fall back to a single
    'unknown' stand-in class covering the whole image, so calibration
    still runs (as global calibration) instead of erroring out."""
    if terrain_mask is not None:
        return terrain_mask
    unknown_class_id = next(
        (cid for cid, name in CLASS_NAMES.items() if "unknown" in name.lower()),
        0
    )
    print(f"No terrain mask provided — using stand-in class {unknown_class_id} "
          f"({CLASS_NAMES.get(unknown_class_id, 'unlabeled')}) for the whole image. "
          f"Calibration will run as GLOBAL, not terrain-aware.")
    return np.full(depth_shape, unknown_class_id, dtype=np.uint8)


def run_before_during(depth_before, srtm_path_before, depth_after, srtm_path_after,
                       terrain_mask_before=None, terrain_mask_after=None):
    terrain_mask_before = _resolve_mask(depth_before.shape, terrain_mask_before)
    terrain_mask_after = _resolve_mask(depth_after.shape, terrain_mask_after)

    elev_before, meta_before = generate_elevation(
        depth_before, srtm_path=srtm_path_before, terrain_mask=terrain_mask_before
    )
    conf_before = compute_confidence(
        terrain_mask_before, meta_before["fit_report"], CLASS_NAMES
    )

    elev_after, meta_after = generate_elevation(
        depth_after, srtm_path=srtm_path_after, terrain_mask=terrain_mask_after
    )
    conf_after = compute_confidence(
        terrain_mask_after, meta_after["fit_report"], CLASS_NAMES
    )

    diff = compute_difference_grid(
        elev_before, elev_after,
        terrain_mask_before, terrain_mask_after,
        conf_before, conf_after
    )

    return {
        "elevation_before": elev_before,
        "elevation_after": elev_after,
        "confidence_before": conf_before,
        "confidence_after": conf_after,
        "diff_grid": diff,
    }


def save_for_person4(result, sample_id, out_dir="person2_elevation/diff_output"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        Path(out_dir) / f"{sample_id}.npz",
        elevation_before=result["elevation_before"],
        elevation_after=result["elevation_after"],
        confidence_before=result["confidence_before"],
        confidence_after=result["confidence_after"],
        elevation_change=result["diff_grid"]["elevation_change"],
        class_changed=result["diff_grid"]["class_changed"],
        combined_confidence=result["diff_grid"]["combined_confidence"],
        significant_change=result["diff_grid"]["significant_change"],
    )
    print(f"Saved {sample_id}.npz to {out_dir}")


if __name__ == "__main__":
    # ============ just change these lines per new pair ============
    DEPTH_BEFORE = "person2_elevation/santa-rosa-wildfire_00000342_pre_disaster_depth.npy"
    SRTM_BEFORE = "person2_elevation/srtm_santa-rosa-wildfire_00000342_pre_disaster_aligned.tif"
    DEPTH_AFTER = "person2_elevation/santa-rosa-wildfire_00000342_post_disaster_depth.npy"
    SRTM_AFTER = "person2_elevation/srtm_santa-rosa-wildfire_00000342_pre_disaster_aligned.tif"
    TERRAIN_MASK_BEFORE = np.load("person2_elevation/santa-rosa-wildfire_00000342_pre_disaster_terrain_mask.npy")   # swap for a real mask path + np.load() once you have one
    TERRAIN_MASK_AFTER = np.load("person2_elevation/santa-rosa-wildfire_00000342_post_disaster_terrain_mask.npy")
    SAMPLE_ID = "SantaRosaWildfire"
    # =================================================================

    depth_before = np.load(DEPTH_BEFORE).astype(np.float32)
    depth_after = np.load(DEPTH_AFTER).astype(np.float32)

    result = run_before_during(
        depth_before, SRTM_BEFORE, depth_after, SRTM_AFTER,
        terrain_mask_before=TERRAIN_MASK_BEFORE,
        terrain_mask_after=TERRAIN_MASK_AFTER,
    )
    save_for_person4(result, sample_id=SAMPLE_ID)

    change = result["diff_grid"]["elevation_change"]
    sig = result["diff_grid"]["significant_change"]
    print(f"\nelevation_change: min={change.min():.3f}, max={change.max():.3f}, "
          f"mean_abs={np.abs(change).mean():.3f}")
    print(f"significant_change: {sig.sum()} / {sig.size} pixels flagged "
          f"({100*sig.mean():.3f}%)")