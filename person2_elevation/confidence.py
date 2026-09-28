import numpy as np

# From your GAMUS per-class validation (Test 4, terrain-wise polynomial) —
# update these if you re-run with the recovered 18 samples or a fine-tuned model
CLASS_VALIDATED_CORRELATION = {
    "ground": 0.014, "low_vegetation": 0.042, "building": 0.288,
    "water": 0.445, "road": 0.024, "tree": 0.325,
}

def compute_confidence(terrain_mask, fit_report, class_names, min_pixels=500, max_reliable_pixels=5000):
    """
    Per-pixel confidence in [0, 1], combining two independent signals:
      1. sample_confidence: how many pixels this class's fit was trained on,
         relative to min_pixels (fallback threshold) and max_reliable_pixels
         (where more data stops meaningfully helping).
      2. validation_confidence: this class's actual measured correlation
         against real GAMUS ground truth.
    Combined as their product — a class needs to be BOTH well-sampled AND
    historically accurate to score high confidence; weak on either drags it down.
    """
    confidence = np.zeros_like(terrain_mask, dtype=np.float32)

    for cls in np.unique(terrain_mask):
        cls_int = int(cls)
        _, n_pixels = fit_report[cls_int]

        sample_conf = np.clip((n_pixels - min_pixels) / (max_reliable_pixels - min_pixels), 0.0, 1.0) \
            if n_pixels >= min_pixels else n_pixels / min_pixels * 0.3

        class_name = class_names.get(cls_int, None)
        val_conf = CLASS_VALIDATED_CORRELATION.get(class_name, 0.3)  # unknown class → conservative default

        confidence[terrain_mask == cls] = sample_conf * val_conf

    return confidence


if __name__ == "__main__":
    fake_mask = np.array([[1, 1, 3], [3, 3, 3], [4, 4, 4]])
    fake_report = {1: ((1, 1), 200), 3: ((1, 1), 6000), 4: ((1, 1), 1403)}
    names = {1: "ground", 3: "building", 4: "water"}

    conf = compute_confidence(fake_mask, fake_report, names)
    print("Confidence grid:\n", np.round(conf, 3))
    print("\nExpected: ground (few pixels, weak class) lowest;")
    print("building (many pixels, decent class) highest;")
    print("water (moderate pixels, best-correlation class) noticeably higher than ground despite similar n")