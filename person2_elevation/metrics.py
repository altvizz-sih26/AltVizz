import numpy as np
from scipy.stats import pearsonr


def compute_metrics(predicted, actual):
    """
    predicted, actual: 1D arrays of the SAME held-out test pixels.
    Returns correlation, MAE, RMSE. NaNs are dropped first (a model may
    have skipped a class with too few pixels, see fit_terrain_wise).
    """
    valid = ~np.isnan(predicted) & ~np.isnan(actual)
    predicted, actual = predicted[valid], actual[valid]

    if len(predicted) < 2:
        return {"correlation": np.nan, "mae": np.nan, "rmse": np.nan, "n": len(predicted)}

    corr, _ = pearsonr(predicted, actual)
    mae = np.mean(np.abs(predicted - actual))
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))

    return {"correlation": corr, "mae": mae, "rmse": rmse, "n": len(predicted)}


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