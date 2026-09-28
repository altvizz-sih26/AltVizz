import numpy as np

data = np.load("person2_elevation/diff_output/bare_terrain_1_test.npz")

print("Keys saved:", list(data.keys()))
print()

elev_change = data["elevation_change"]
sig_change = data["significant_change"]
conf_before = data["confidence_before"]
conf_after = data["confidence_after"]

print(f"elevation_change: min={elev_change.min():.4f}, max={elev_change.max():.4f}, mean={np.abs(elev_change).mean():.4f}")
print(f"significant_change: {sig_change.sum()} pixels flagged True out of {sig_change.size} ({100*sig_change.mean():.3f}%)")
print()
print(f"confidence_before range: {conf_before.min():.3f} to {conf_before.max():.3f}")
print(f"confidence_after range:  {conf_after.min():.3f} to {conf_after.max():.3f}")
print()

if np.abs(elev_change).max() < 0.01 and sig_change.sum() == 0:
    print("PASS: before/after identical image -> no real change detected, as expected.")
else:
    print("CHECK THIS: same image was used twice, but the diff grid shows non-trivial change or flagged pixels.")
    print("That would mean something in generate_elevation or the diff logic isn't deterministic, or there's a real bug.")