import numpy as np
import elevation_pipeline as ep

# Two fake terrain classes, DIFFERENT known relationships — same trick as fit_methods.py
rng = np.random.default_rng(0)
depth = rng.uniform(0, 1, size=(100, 100)).astype(np.float32)
terrain_mask = np.where(depth > 0.5, 1, 2)

# class 1: TRUE relationship is a straight line — height = 20*depth + 5
# class 2: TRUE relationship is a real curve — height = 30*depth^2 + 2*depth + 10
true_elev = np.where(
    terrain_mask == 1,
    20 * depth + 5,
    30 * depth**2 + 2 * depth + 10
)
reference_elevation = (true_elev + rng.normal(0, 0.3, size=depth.shape)).astype(np.float32)
void_mask = np.zeros_like(depth, dtype=bool)

print("=== degree=1 (should behave like the OLD linear code) ===")
calibrated_lin, report_lin = ep.terrain_aware_calibration(depth, reference_elevation, void_mask, terrain_mask, degree=1)
for cls, (coeffs, n) in report_lin.items():
    print(f"  class {cls}: coeffs={np.round(coeffs, 2)}  n={n}")
print(f"  class 1 true: a=20, b=5 (should be close)")
print(f"  class 2 true line-only approx (won't fully match — real relationship is curved)")

print("\n=== degree=2 (should recover the REAL curve for class 2) ===")
calibrated_poly, report_poly = ep.terrain_aware_calibration(depth, reference_elevation, void_mask, terrain_mask, degree=2)
for cls, (coeffs, n) in report_poly.items():
    print(f"  class {cls}: coeffs={np.round(coeffs, 2)}  n={n}")
print(f"  class 1 true: a=0(x^2), 20(x), 5  (should show ~0 curve term)")
print(f"  class 2 true: a=30(x^2), 2(x), 10  (should now match closely)")

print("\n=== Prediction sanity check ===")
print(f"  degree=1 elevation range: {calibrated_lin.min():.2f} to {calibrated_lin.max():.2f}")
print(f"  degree=2 elevation range: {calibrated_poly.min():.2f} to {calibrated_poly.max():.2f}")
print(f"  reference range:          {reference_elevation.min():.2f} to {reference_elevation.max():.2f}")