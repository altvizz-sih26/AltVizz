import numpy as np
from pathlib import Path
from load_gamus import load_gamus_sample, get_valid_mask

GAMUS_DIR = Path("gamus_test")
SEED = 42  # fixed seed so the split is reproducible, not different every run

def get_sample_ids():
    return sorted(p.stem.replace("_RGB", "") for p in (GAMUS_DIR / "images").glob("*_RGB.h5"))

def build_split(test_fraction=0.2, seed=SEED):
    """
    For every GAMUS sample, finds valid pixels (via get_valid_mask) and
    randomly assigns each one to train or test, per-sample, so every
    sample contributes to both sets rather than some samples being
    entirely train and others entirely test.
    Returns a dict: sample_id -> (train_mask, test_mask), both boolean
    arrays same shape as the image, safe to reuse across all 4 experiments.
    """
    rng = np.random.default_rng(seed)
    splits = {}

    for sample_id in get_sample_ids():
        rgb, agl, class_map = load_gamus_sample(sample_id)
        valid = get_valid_mask(agl, class_map)

        rand_vals = rng.random(valid.shape)
        is_test = valid & (rand_vals < test_fraction)
        is_train = valid & ~is_test

        splits[sample_id] = {"train": is_train, "test": is_test}

    return splits

if __name__ == "__main__":
    splits = build_split()
    total_train = sum(s["train"].sum() for s in splits.values())
    total_test = sum(s["test"].sum() for s in splits.values())
    print(f"Samples processed: {len(splits)}")
    print(f"Total train pixels: {total_train}")
    print(f"Total test pixels: {total_test}")
    print(f"Actual test fraction: {total_test / (total_train + total_test):.3f}")

    np.savez("person2_elevation/gamus_split.npz",
             **{f"{sid}_train": s["train"] for sid, s in splits.items()},
             **{f"{sid}_test": s["test"] for sid, s in splits.items()})
    print("Saved to person2_elevation/gamus_split.npz")