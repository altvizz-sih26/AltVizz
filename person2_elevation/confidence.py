"""
Confidence map: how much to trust the elevation value at each pixel.

confidence(pixel) = sample_confidence(class) x validation_confidence(class)

- sample_confidence: how many pixels of this class exist in THIS image.
- validation_confidence: derived from the held-out MAE measured for each
  classifier class.

Lower validation MAE = higher validation confidence.

The confidence values are always in the range 0.0 to 1.0.
"""

import numpy as np


# Held-out MAE from the current fine-tuned validation run.
# These are elevation errors in metres.
CLASS_VALIDATED_MAE = {
    "urban": 7.750,
    "vegetation": 4.233,
    "bare_terrain": 3.908,
    "shadow": 0.935,
    "unknown_low_confidence": 3.000,
}

# Water had only 24 held-out pixels in the current validation,
# so its MAE is not considered reliable enough to use directly.
# It therefore falls back to DEFAULT_VALIDATION.
DEFAULT_VALIDATION = 0.3

# Reference error scale in metres.
# An MAE of 10 m corresponds to a validation confidence of 0.5.
VALIDATION_ERROR_SCALE = 10.0

_warned = set()


def _validation_confidence(mae):
    """
    Converts validation MAE into a 0-1 confidence score.

    Lower MAE -> higher confidence.
    """
    confidence = 1.0 / (1.0 + mae / VALIDATION_ERROR_SCALE)
    return float(np.clip(confidence, 0.0, 1.0))


def _sample_confidence(n, min_pixels, max_reliable_pixels):
    """
    Confidence based on how many pixels of this terrain class
    are present in the current image.
    """
    if n <= 0:
        return 0.0

    if n < min_pixels:
        return 0.3 * n / min_pixels

    if n >= max_reliable_pixels:
        return 1.0

    return 0.3 + 0.7 * (
        (n - min_pixels) / (max_reliable_pixels - min_pixels)
    )


def compute_confidence(
    terrain_mask,
    fit_report=None,
    class_names=None,
    min_pixels=500,
    max_reliable_pixels=5000,
):
    """
    Computes a 0-1 confidence map.

    fit_report is accepted for backward compatibility but is not used.

    Final confidence =
        sample confidence × validation confidence
    """

    conf = np.zeros(terrain_mask.shape, dtype=np.float32)

    for class_id in np.unique(terrain_mask):

        in_class = terrain_mask == class_id

        name = (class_names or {}).get(
            int(class_id),
            f"class_{int(class_id)}"
        )

        mae = CLASS_VALIDATED_MAE.get(name)

        if mae is None:
            validation = DEFAULT_VALIDATION

            if name not in _warned:
                print(
                    f"[confidence] no validated MAE for class '{name}' "
                    f"- using fallback {DEFAULT_VALIDATION}"
                )
                _warned.add(name)

        else:
            validation = _validation_confidence(mae)

            print(
                f"[confidence] {name}: "
                f"MAE={mae:.3f} m -> validation confidence={validation:.3f}"
            )

        sample = _sample_confidence(
            int(in_class.sum()),
            min_pixels,
            max_reliable_pixels
        )

        conf[in_class] = sample * validation

    return np.clip(conf, 0.0, 1.0)