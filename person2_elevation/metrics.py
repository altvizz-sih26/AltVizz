import numpy as np
from scipy.stats import pearsonr


def _f(x):
    """Native Python float, or None for NaN/inf (JSON-safe)."""
    x = float(x)
    return x if np.isfinite(x) else None


def compute_metrics(predicted, actual):
    """
    predicted, actual: 1D arrays of the SAME held-out test pixels.
    Returns correlation, MAE, RMSE as native Python floats (None if undefined)
    so the result can go straight into json.dumps. NaNs are dropped first
    (a model may have skipped a class with too few pixels, see fit_terrain_wise).
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    valid = np.isfinite(predicted) & np.isfinite(actual)
    predicted, actual = predicted[valid], actual[valid]
    n = int(len(predicted))

    if n < 2:
        return {"correlation": None, "mae": None, "rmse": None, "n": n}

    # Pearson is undefined if either side is constant (e.g. class-mean baseline)
    if np.std(predicted) == 0 or np.std(actual) == 0:
        corr = None
    else:
        corr = _f(pearsonr(predicted, actual)[0])

    mae = _f(np.mean(np.abs(predicted - actual)))
    rmse = _f(np.sqrt(np.mean((predicted - actual) ** 2)))

    return {"correlation": corr, "mae": mae, "rmse": rmse, "n": n}


if __name__ == "__main__":
    # --- Sanity test 1: perfect prediction, should give corr=1, mae=0, rmse=0 ---
    actual = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    predicted_perfect = actual.copy()
    print("=== Perfect prediction (expect corr≈1.0, mae≈0, rmse≈0) ===")
    print(compute_metrics(predicted_perfect, actual))

    # --- Sanity test 2: known, deliberate error ---
    predicted_off = actual + 2.0  # every point off by exactly 2
    print("\n=== Constant +2 error (expect corr≈1.0, mae=2.0, rmse=2.0) ===")
    print(compute_metrics(predicted_off, actual))

    # --- Sanity test 3: NaNs should be dropped, not crash or corrupt the result ---
    predicted_with_nan = np.array([5.0, np.nan, 15.0, 20.0, 25.0])
    print("\n=== With one NaN (expect n=4, same corr/mae/rmse as if it were removed) ===")
    print(compute_metrics(predicted_with_nan, actual))

    # --- Sanity test 4: constant prediction, corr should be None not a crash ---
    print("\n=== Constant prediction (expect corr=None) ===")
    print(compute_metrics(np.full(5, 12.0), actual))

    # --- Sanity test 5: float32 input must still be JSON-serialisable ---
    import json
    print("\n=== float32 input, json.dumps must work ===")
    print(json.dumps(compute_metrics(predicted_off.astype(np.float32), actual.astype(np.float32))))