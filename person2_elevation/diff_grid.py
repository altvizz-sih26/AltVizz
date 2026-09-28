import numpy as np


def compute_difference_grid(elev_before, elev_after, class_before, class_after,
                             conf_before, conf_after, max_margin=2.0):
    """
    All inputs: same-shape 2D arrays, already aligned (Person 3's job).
    max_margin: largest noise allowance at zero confidence, in meters —
    tune this once real confidence values exist.
    """
    elev_before = np.asarray(elev_before, dtype=np.float32)
    elev_after = np.asarray(elev_after, dtype=np.float32)

    if elev_before.shape != elev_after.shape:
        raise ValueError(f"Shape mismatch: before={elev_before.shape}, after={elev_after.shape}")
    if class_before.shape != elev_before.shape or class_after.shape != elev_before.shape:
        raise ValueError("class arrays must match elevation grid shape")
    if conf_before.shape != elev_before.shape or conf_after.shape != elev_before.shape:
        raise ValueError("confidence arrays must match elevation grid shape")

    elevation_change = elev_after - elev_before
    class_changed = class_before != class_after
    combined_confidence = np.minimum(conf_before, conf_after)
    margin = (1.0 - combined_confidence) * max_margin
    significant_change = np.abs(elevation_change) > margin

    return {
        "elevation_change": elevation_change,
        "class_changed": class_changed,
        "combined_confidence": combined_confidence,
        "margin": margin,
        "significant_change": significant_change,
    }


def summarize_difference_grid(diff_grid):
    sig = diff_grid["significant_change"]
    return {
        "total_points": sig.size,
        "significant_points": int(sig.sum()),
        "significant_pct": float(100 * sig.sum() / sig.size),
        "class_changed_points": int(diff_grid["class_changed"].sum()),
        "mean_abs_change_where_significant": float(
            np.abs(diff_grid["elevation_change"][sig]).mean()
        ) if sig.sum() > 0 else 0.0,
    }