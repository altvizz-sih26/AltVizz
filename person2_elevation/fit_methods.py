import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures


def fit_global_linear(depth, target):
    """Test 1: one linear fit over all pixels, no terrain classes."""
    X = depth.reshape(-1, 1)
    y = target.reshape(-1)
    model = LinearRegression().fit(X, y)
    return model


def fit_terrain_wise(depth, target, class_map, model_type="linear", degree=1):
    """
    Tests 2-4: one fit per terrain class.
    model_type: "linear", "ransac", "huber"
    degree: 1 for linear/robust, 2 or 3 for polynomial (only valid with model_type="linear")
    Returns a dict: class_id -> fitted model (plus a "poly" transformer if degree > 1)
    """
    models = {}
    for class_id in np.unique(class_map):
        cls_pixels = class_map == class_id
        X = depth[cls_pixels].reshape(-1, 1)
        y = target[cls_pixels]

        if len(y) < 10:
            print(f"  class {class_id}: only {len(y)} pixels, skipping (too few to fit)")
            continue

        poly = None
        if degree > 1:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X = poly.fit_transform(X)

        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "ransac":
            model = RANSACRegressor(LinearRegression(), random_state=42)
        elif model_type == "huber":
            model = HuberRegressor()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X, y)
        models[class_id] = {"model": model, "poly": poly, "n_pixels": len(y)}

    return models


def predict_terrain_wise(depth, class_map, models):
    """Applies the right per-class model to every pixel, returns a full prediction array."""
    pred = np.full(depth.shape, np.nan, dtype=np.float32)
    for class_id, entry in models.items():
        cls_pixels = class_map == class_id
        X = depth[cls_pixels].reshape(-1, 1)
        if entry["poly"] is not None:
            X = entry["poly"].transform(X)
        pred[cls_pixels] = entry["model"].predict(X)
    return pred


if __name__ == "__main__":
    # --- Sanity test with synthetic data, known ground truth ---
    rng = np.random.default_rng(0)
    depth = rng.uniform(0, 1, size=(100, 100)).astype(np.float32)

    # Two fake classes with DIFFERENT known relationships:
    # class 1: height = 20*depth + 5
    # class 2: height = 10*depth + 50
    class_map = np.where(depth > 0.5, 1, 2)
    true_height = np.where(class_map == 1, 20 * depth + 5, 10 * depth + 50)
    noisy_height = true_height + rng.normal(0, 0.5, size=depth.shape)  # small noise

    print("=== Test 1: Global linear (should be a poor/blended fit, that's expected) ===")
    g_model = fit_global_linear(depth, noisy_height)
    print(f"  a={g_model.coef_[0]:.2f}, b={g_model.intercept_:.2f}  (blends both classes, won't match either)")

    print("\n=== Test 2: Terrain-wise linear (should recover ~20,5 and ~10,50) ===")
    tw_models = fit_terrain_wise(depth, noisy_height, class_map, model_type="linear", degree=1)
    for cls, entry in tw_models.items():
        m = entry["model"]
        print(f"  class {cls}: a={m.coef_[0]:.2f}, b={m.intercept_:.2f}  (n={entry['n_pixels']})")

    print("\n=== Test 3: Terrain-wise Huber (should also recover ~20,5 and ~10,50) ===")
    huber_models = fit_terrain_wise(depth, noisy_height, class_map, model_type="huber", degree=1)
    for cls, entry in huber_models.items():
        m = entry["model"]
        print(f"  class {cls}: a={m.coef_[0]:.2f}, b={m.intercept_:.2f}  (n={entry['n_pixels']})")

    print("\n=== Test 4: Terrain-wise polynomial degree 2 (should still recover ~linear coefficients, tiny curve term) ===")
    poly_models = fit_terrain_wise(depth, noisy_height, class_map, model_type="linear", degree=2)
    for cls, entry in poly_models.items():
        m = entry["model"]
        print(f"  class {cls}: coefs={m.coef_}, intercept={m.intercept_:.2f}  (n={entry['n_pixels']})")