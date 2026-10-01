"""
Before/during pipeline for Person 1's fine-tuned height outputs.

Person 1 delivers, per image:
    <name>_height_normalized.npy   0-1, min-max scaled   -> SRTM mode
    <name>_height_meters.npy       metres, may be < 0    -> no-SRTM mode
    <name>_terrain_mask.npy        class ids 0-5

Copy those into DATA_DIR (infer.py writes to ./height_predictions),
edit the CONFIG block, run:
    python person2_elevation/run_before_during_pipeline.py
"""
from pathlib import Path
import json
import numpy as np
import rasterio

from elevation_pipeline import generate_elevation, relative_elevation, CLASS_NAMES
from confidence import compute_confidence
from diff_grid import compute_difference_grid
from fit_methods import fit_terrain_wise, predict_terrain_wise
from metrics import compute_metrics

# ============ CONFIG - change per pair ============
DATA_DIR = "person2_elevation"
PRE_NAME = "santa-rosa-wildfire_00000342_pre_disaster"
POST_NAME = "santa-rosa-wildfire_00000342_post_disaster"
SRTM_PATH = "person2_elevation/srtm_santa-rosa-wildfire_00000342_pre_disaster_aligned.tif"  # same for pre and post
USE_SRTM = True        # True: height_normalized + SRTM calibration. False: height_meters used directly.
RUN_SUFFIX = "finetuned" # keeps vanilla / fine-tuned runs from overwriting each other
NOISE_K = 3.0            # change must exceed NOISE_K x measured noise to count as real
HOLDOUT_BLOCK = 64
HOLDOUT_FRACTION = 0.2
HOLDOUT_SEED = 42
HOLDOUT_DEGREE = 2       # try 1 as well: on a single image it is often more stable
MIN_CLASS_PIXELS = 10    # classes with fewer training pixels are skipped
# ==================================================


def _fmt(x):
    """Format a metric that may be None (undefined)."""
    return "nan" if x is None else f"{x:.3f}"


def _to_native(o):
    """Recursively convert numpy types to plain Python so json.dumps never fails.
    NaN/inf become None (they are not valid JSON)."""
    if isinstance(o, dict):
        return {str(k): _to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_native(v) for v in o]
    if isinstance(o, np.ndarray):
        return _to_native(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def load_height(name, use_srtm):
    kind = "height_normalized" if use_srtm else "height_meters"
    path = Path(DATA_DIR) / f"{name}_{kind}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - copy Person 1's output into {DATA_DIR}")
    arr = np.load(path).astype(np.float32)
    if use_srtm and (np.nanmin(arr) < -0.01 or np.nanmax(arr) > 1.01):
        print(f"WARNING: {path.name} is outside 0-1; is this really height_normalized?")
    return arr


def load_mask(name):
    path = Path(DATA_DIR) / f"{name}_terrain_mask.npy"
    return np.load(path) if path.exists() else None


def _resolve_mask(shape, mask):
    if mask is not None:
        return mask
    uid = next((c for c, n in CLASS_NAMES.items() if "unknown" in n.lower()), 0)
    print(f"No terrain mask - using single stand-in class {uid}; calibration is GLOBAL, not terrain-aware.")
    return np.full(shape, uid, dtype=np.uint8)


def block_holdout_mask(shape, block=HOLDOUT_BLOCK, fraction=HOLDOUT_FRACTION, seed=HOLDOUT_SEED):
    rng = np.random.default_rng(seed)
    nby, nbx = -(-shape[0] // block), -(-shape[1] // block)
    chosen = rng.random((nby, nbx)) < fraction
    return np.kron(chosen, np.ones((block, block), dtype=bool))[:shape[0], :shape[1]]


def compute_holdout_metrics(height_normalized, srtm_path, terrain_mask, label):
    """Fit terrain-wise degree-2 polynomial calibration on training blocks
    and evaluate only on unseen held-out blocks.
    """

    with rasterio.open(srtm_path) as ds:
        ref = ds.read(1).astype(np.float32)
        nodata = ds.nodata

    if ref.shape != height_normalized.shape:
        print(
            f"[{label}] SRTM shape {ref.shape} != height shape "
            f"{height_normalized.shape}, skipping held-out metrics"
        )
        return None

    valid_srtm = np.isfinite(ref)

    if nodata is not None:
        valid_srtm &= ref != nodata

    valid_srtm &= (ref != -32768) & (ref >= -1000)

    valid = np.isfinite(height_normalized) & valid_srtm

    is_test = valid & block_holdout_mask(height_normalized.shape)
    is_train = valid & ~is_test

    if is_train.sum() < 500 or is_test.sum() < 100:
        print(
            f"[{label}] too few valid pixels "
            f"({is_train.sum()} train / {is_test.sum()} test), "
            f"skipping held-out metrics"
        )
        return None

    x_tr = height_normalized[is_train]
    y_tr = ref[is_train]
    c_tr = terrain_mask[is_train]

    x_te = height_normalized[is_test].copy()
    c_te = terrain_mask[is_test]
    truth = ref[is_test]

    y_bounds = {}
    class_mean = {}

    for cid in np.unique(c_tr):
        s = c_tr == cid

        if s.sum() < MIN_CLASS_PIXELS:
            continue

        lo, hi = np.percentile(x_tr[s], [1, 99])

        sel = c_te == cid
        x_te[sel] = np.clip(x_te[sel], lo, hi)

        y_bounds[int(cid)] = (
            float(y_tr[s].min()),
            float(y_tr[s].max())
        )

        class_mean[int(cid)] = float(y_tr[s].mean())

    fit = fit_terrain_wise(
        x_tr,
        y_tr,
        c_tr,
        model_type="linear",
        degree=HOLDOUT_DEGREE
    )

    pred = np.asarray(
        predict_terrain_wise(x_te, c_te, fit),
        dtype=np.float64
    ).copy()

    n_clamped = 0
    base = np.full(pred.shape, np.nan, dtype=np.float64)

    for cid, (ylo, yhi) in y_bounds.items():
        sel = c_te == cid

        before = pred[sel]
        clipped = np.clip(before, ylo, yhi)

        n_clamped += int(
            np.sum(
                np.isfinite(before) &
                (before != clipped)
            )
        )

        pred[sel] = clipped
        base[sel] = class_mean[cid]

    print(
        f"[{label}] {n_clamped}/{pred.size} predictions hit "
        f"the physical clamp"
    )

    pooled = compute_metrics(pred, truth)
    baseline = compute_metrics(base, truth)

    per_class = {}
    per_class_baseline = {}

    for cid in np.unique(c_te):
        sel = c_te == cid

        if sel.sum() < MIN_CLASS_PIXELS:
            continue

        name = CLASS_NAMES.get(
            int(cid),
            f"class_{int(cid)}"
        )

        per_class[name] = compute_metrics(
            pred[sel],
            truth[sel]
        )

        per_class_baseline[name] = compute_metrics(
            base[sel],
            truth[sel]
        )

    print(
        f"[{label}] HELD-OUT pooled: "
        f"corr={_fmt(pooled['correlation'])} "
        f"mae={_fmt(pooled['mae'])} "
        f"rmse={_fmt(pooled['rmse'])} "
        f"n={pooled['n']}"
    )

    print(
        f"[{label}] BASELINE (class mean): "
        f"mae={_fmt(baseline['mae'])} "
        f"rmse={_fmt(baseline['rmse'])}"
    )

    for name, m in per_class.items():
        b = per_class_baseline[name]

        print(
            f"    {name:24s}: "
            f"corr={_fmt(m['correlation'])} "
            f"mae={_fmt(m['mae'])} "
            f"rmse={_fmt(m['rmse'])} "
            f"n={m['n']} "
            f"(baseline mae={_fmt(b['mae'])})"
        )

    return {
        "pooled": pooled,
        "per_class": per_class,
        "baseline_class_mean": baseline,
        "baseline_per_class": per_class_baseline,
        "n_clamped": n_clamped
    }
    """Fit terrain-wise degree-2 polynomial calibration on training blocks
    and evaluate only on unseen held-out blocks.
    """

    with rasterio.open(srtm_path) as ds:
        ref = ds.read(1).astype(np.float32)
        nodata = ds.nodata

    if ref.shape != height_normalized.shape:
        print(
            f"[{label}] SRTM shape {ref.shape} != height shape "
            f"{height_normalized.shape}, skipping held-out metrics"
        )
        return None

    # Exclude invalid / NoData SRTM pixels
    valid_srtm = np.isfinite(ref)

    if nodata is not None:
        valid_srtm &= ref != nodata

    valid_srtm &= (ref != -32768) & (ref >= -1000)

    valid = np.isfinite(height_normalized) & valid_srtm

    # Spatial block holdout
    is_test = valid & block_holdout_mask(height_normalized.shape)
    is_train = valid & ~is_test

    if is_train.sum() < 500 or is_test.sum() < 100:
        print(
            f"[{label}] too few valid pixels "
            f"({is_train.sum()} train / {is_test.sum()} test), "
            f"skipping held-out metrics"
        )
        return None

    x_tr = height_normalized[is_train]
    y_tr = ref[is_train]
    c_tr = terrain_mask[is_train]

    x_te = height_normalized[is_test].copy()
    c_te = terrain_mask[is_test]
    truth = ref[is_test]

    # Per-class training ranges
    y_bounds = {}
    class_mean = {}

    for cid in np.unique(c_tr):
        s = c_tr == cid

        if s.sum() < MIN_CLASS_PIXELS:
            continue

        lo, hi = np.percentile(x_tr[s], [1, 99])

        sel = c_te == cid
        x_te[sel] = np.clip(x_te[sel], lo, hi)

        y_bounds[int(cid)] = (
            float(y_tr[s].min()),
            float(y_tr[s].max())
        )

        class_mean[int(cid)] = float(y_tr[s].mean())

    # Same calibration method as production:
    # terrain-wise polynomial, degree 2
    fit = fit_terrain_wise(
        x_tr,
        y_tr,
        c_tr,
        model_type="linear",
        degree=HOLDOUT_DEGREE
    )

    pred = np.asarray(
        predict_terrain_wise(x_te, c_te, fit),
        dtype=np.float64
    ).copy()

    # Prevent unstable polynomial extrapolation
    n_clamped = 0

    base = np.full(pred.shape, np.nan, dtype=np.float64)

    for cid, (ylo, yhi) in y_bounds.items():
        sel = c_te == cid

        before = pred[sel]
        clipped = np.clip(before, ylo, yhi)

        n_clamped += int(
            np.sum(
                np.isfinite(before) &
                (before != clipped)
            )
        )

        pred[sel] = clipped
        base[sel] = class_mean[cid]

    print(
        f"[{label}] {n_clamped}/{pred.size} predictions hit "
        f"the physical clamp"
    )

    # Overall held-out metrics
    pooled = compute_metrics(pred, truth)

    # Simple per-class mean baseline
    baseline = compute_metrics(base, truth)

    per_class = {}
    per_class_baseline = {}

    for cid in np.unique(c_te):
        sel = c_te == cid

        if sel.sum() < MIN_CLASS_PIXELS:
            continue

        name = CLASS_NAMES.get(
            int(cid),
            f"class_{int(cid)}"
        )

        per_class[name] = compute_metrics(
            pred[sel],
            truth[sel]
        )

        per_class_baseline[name] = compute_metrics(
            base[sel],
            truth[sel]
        )

    print(
        f"[{label}] HELD-OUT pooled: "
        f"corr={_fmt(pooled['correlation'])} "
        f"mae={_fmt(pooled['mae'])} "
        f"rmse={_fmt(pooled['rmse'])} "
        f"n={pooled['n']}"
    )

    print(
        f"[{label}] BASELINE (class mean): "
        f"mae={_fmt(baseline['mae'])} "
        f"rmse={_fmt(baseline['rmse'])}"
    )

    for name, m in per_class.items():
        b = per_class_baseline[name]

        print(
            f"    {name:24s}: "
            f"corr={_fmt(m['correlation'])} "
            f"mae={_fmt(m['mae'])} "
            f"rmse={_fmt(m['rmse'])} "
            f"n={m['n']} "
            f"(baseline mae={_fmt(b['mae'])})"
        )

    return {
        "pooled": pooled,
        "per_class": per_class,
        "baseline_class_mean": baseline,
        "baseline_per_class": per_class_baseline,
        "n_clamped": n_clamped
    }
    """Fit on most of the image, score ONLY on held-out blocks it never saw.
    This is the honest number - separate from the in-sample fit used for
    the actual delivered elevation. Returns None if there isn't enough
    valid data to do this meaningfully.

    Guards against polynomial extrapolation blow-up:
      1. held-out inputs are clipped, per class, to the 1st-99th percentile
         range of that class's TRAINING inputs;
      2. predictions are clipped, per class, to the min/max of that class's
         training SRTM values (safety net);
      3. a per-class-mean baseline is reported so you can see whether the
         calibration beats a trivial predictor.
    """
    with rasterio.open(srtm_path) as ds:
         ref = ds.read(1).astype(np.float32)
         nodata = ds.nodata

    if nodata is not None:
        valid_srtm = ref != nodata
    else:
        valid_srtm = np.ones(ref.shape, dtype=bool)

# Match elevation_pipeline.py's invalid-elevation rules
    valid_srtm &= np.isfinite(ref) & (ref != -32768) & (ref >= -1000)

    valid = np.isfinite(height_normalized) & valid_srtm
    is_test = valid & block_holdout_mask(height_normalized.shape)
    is_train = valid & ~is_test
    if is_train.sum() < 500 or is_test.sum() < 100:
        print(f"[{label}] too few valid pixels ({is_train.sum()} train / {is_test.sum()} test), skipping held-out metrics")
        return None

    x_tr, y_tr, c_tr = height_normalized[is_train], ref[is_train], terrain_mask[is_train]
    x_te, c_te = height_normalized[is_test].copy(), terrain_mask[is_test]
    truth = ref[is_test]

    # Per-class ranges learned from TRAINING pixels only
    y_bounds, class_mean = {}, {}
    for cid in np.unique(c_tr):
        s = c_tr == cid
        if s.sum() < MIN_CLASS_PIXELS:
            continue
        lo, hi = np.percentile(x_tr[s], [1, 99])
        sel = c_te == cid
        x_te[sel] = np.clip(x_te[sel], lo, hi)          # never evaluate outside trained range
        y_bounds[int(cid)] = (float(y_tr[s].min()), float(y_tr[s].max()))
        class_mean[int(cid)] = float(y_tr[s].mean())

    fit = fit_terrain_wise(x_tr, y_tr, c_tr, model_type="linear", degree=HOLDOUT_DEGREE)
    pred = np.asarray(predict_terrain_wise(x_te, c_te, fit), dtype=np.float64).copy()

    # Safety net + baseline
    n_clamped = 0
    base = np.full(pred.shape, np.nan, dtype=np.float64)
    for cid, (ylo, yhi) in y_bounds.items():
        sel = c_te == cid
        before = pred[sel]
        clipped = np.clip(before, ylo, yhi)
        n_clamped += int(np.sum(np.isfinite(before) & (before != clipped)))
        pred[sel] = clipped
        base[sel] = class_mean[cid]
    print(f"[{label}] {n_clamped}/{pred.size} predictions hit the physical clamp "
          f"(high = the fit is unstable, not just unlucky)")

    pooled = compute_metrics(pred, truth)
    baseline = compute_metrics(base, truth)   # predict per-class training mean
    per_class, per_class_baseline = {}, {}
    for cid in np.unique(c_te):
        sel = c_te == cid
        if sel.sum() < MIN_CLASS_PIXELS:
            continue
        name = CLASS_NAMES.get(int(cid), f"class_{int(cid)}")
        per_class[name] = compute_metrics(pred[sel], truth[sel])
        per_class_baseline[name] = compute_metrics(base[sel], truth[sel])

    print(f"[{label}] HELD-OUT pooled: corr={_fmt(pooled['correlation'])} "
          f"mae={_fmt(pooled['mae'])} rmse={_fmt(pooled['rmse'])} n={pooled['n']}")
    print(f"[{label}] BASELINE (class mean): mae={_fmt(baseline['mae'])} rmse={_fmt(baseline['rmse'])}")
    for name, m in per_class.items():
        b = per_class_baseline[name]
        print(f"    {name:24s}: corr={_fmt(m['correlation'])} mae={_fmt(m['mae'])} "
              f"rmse={_fmt(m['rmse'])} n={m['n']}  (baseline mae={_fmt(b['mae'])})")

    return {"pooled": pooled, "per_class": per_class,
            "baseline_class_mean": baseline,
            "baseline_per_class": per_class_baseline,
            "n_clamped": n_clamped}


def _elevation(height, srtm_path, mask, use_srtm, label):
    if use_srtm:
        elev, meta = generate_elevation(height, srtm_path=srtm_path, terrain_mask=mask)
        print(f"[{label}] in-sample vs SRTM (fit and scored on the same pixels - optimistic): "
              f"corr={meta.get('correlation')}, mean err={meta.get('error_mean')}")
        return elev
    return relative_elevation(np.asarray(height, dtype=np.float32))


def apply_noise_floor(diff, k=NOISE_K):
    change = diff["elevation_change"]
    stable = (~np.asarray(diff["class_changed"]).astype(bool)) & np.isfinite(change)
    if stable.sum() < 1000:
        stable = np.isfinite(change)
    vals = change[stable]
    shift = float(np.median(vals))
    sigma = 1.4826 * float(np.median(np.abs(vals - shift)))
    floor = k * sigma
    sig = np.asarray(diff["significant_change"]).astype(bool) & (np.abs(change) > floor)
    return sig, floor, sigma, shift


def run_before_during(h_before, h_after, mask_before=None, mask_after=None,
                      srtm_path=None, use_srtm=True):
    mask_before = _resolve_mask(h_before.shape, mask_before)
    mask_after = _resolve_mask(h_after.shape, mask_after)
    shapes = {"height_before": h_before.shape, "height_after": h_after.shape,
              "mask_before": mask_before.shape, "mask_after": mask_after.shape}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Before/after arrays must share one shape, got {shapes}")

    elev_b = _elevation(h_before, srtm_path, mask_before, use_srtm, "before")
    elev_a = _elevation(h_after, srtm_path, mask_after, use_srtm, "after")
    conf_b = compute_confidence(mask_before, class_names=CLASS_NAMES)
    conf_a = compute_confidence(mask_after, class_names=CLASS_NAMES)

    diff = compute_difference_grid(elev_b, elev_a, mask_before, mask_after, conf_b, conf_a)
    sig, floor, sigma, shift = apply_noise_floor(diff)
    diff["significant_change"] = sig
    diff["noise_floor"] = floor
    print(f"Measured noise sigma={sigma:.3f} m, floor={floor:.3f} m, "
          f"median before/after shift={shift:.3f} m")

    metrics_b = metrics_a = None
    if use_srtm:
        metrics_b = compute_holdout_metrics(h_before, srtm_path, mask_before, "before")
        metrics_a = compute_holdout_metrics(h_after, srtm_path, mask_after, "after")
    else:
        print("meters mode: no per-image SRTM ground truth to score against - "
              "accuracy for this path can only come from offline GAMUS validation, "
              "not from a real deployed image. metrics_before/after will be null.")

    return {"elevation_before": elev_b, "elevation_after": elev_a,
            "confidence_before": conf_b, "confidence_after": conf_a,
            "diff_grid": diff, "mode": "srtm_calibrated" if use_srtm else "height_above_ground_m",
            "metrics_before": metrics_b, "metrics_after": metrics_a}


def save_for_person4(result, sample_id, out_dir="person2_elevation/diff_output"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    d = result["diff_grid"]
    np.savez(Path(out_dir) / f"{sample_id}.npz",
             elevation_before=result["elevation_before"],
             elevation_after=result["elevation_after"],
             confidence_before=result["confidence_before"],
             confidence_after=result["confidence_after"],
             elevation_change=d["elevation_change"],
             class_changed=d["class_changed"],
             combined_confidence=d["combined_confidence"],
             significant_change=d["significant_change"],
             noise_floor=np.float32(d["noise_floor"]),
             mode=np.array(result["mode"]),
             # metrics saved as JSON text, since they're nested per-class dicts,
             # not a fixed-shape array. Load back with:
             #   json.loads(str(np.load(path)["metrics_before"]))
             # _to_native guarantees numpy scalars / NaN never break json.dumps.
             metrics_before=np.array(json.dumps(_to_native(result["metrics_before"]))),
             metrics_after=np.array(json.dumps(_to_native(result["metrics_after"]))))
    print(f"Saved {sample_id}.npz to {out_dir}")


if __name__ == "__main__":
    h_b = load_height(PRE_NAME, USE_SRTM)
    h_a = load_height(POST_NAME, USE_SRTM)
    result = run_before_during(h_b, h_a, load_mask(PRE_NAME), load_mask(POST_NAME),
                               srtm_path=SRTM_PATH, use_srtm=USE_SRTM)
    tag = "srtm" if USE_SRTM else "meters"
    save_for_person4(result, f"{PRE_NAME.replace('_pre_disaster', '')}_{tag}_{RUN_SUFFIX}")
    sig = result["diff_grid"]["significant_change"]
    print(f"significant_change: {sig.sum()} / {sig.size} pixels ({100 * sig.mean():.3f}%)")