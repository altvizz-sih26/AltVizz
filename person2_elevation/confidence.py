"""
Confidence map: how much to trust the elevation value at each pixel.

confidence(pixel) = sample_confidence(class) x validation_confidence(class)

- sample_confidence: how many pixels of this class exist in THIS image
  (counted straight from the terrain mask, so it works in SRTM mode and
  no-SRTM mode alike - no fit_report needed).
- validation_confidence: how well this class's height estimate correlated
  with real ground truth when it was tested. Keyed by the classifier's OWN
  class names (urban, vegetation, bare_terrain, water, shadow,
  unknown_low_confidence) - NOT the GAMUS labels.

CLASS_VALIDATED_CORRELATION is empty on purpose: the old numbers were for
vanilla DA-V2 on GAMUS classes and did not match these names, so they were
silently replaced by the default anyway. Fill this in only with numbers
measured for the fine-tuned model, per classifier class.
"""
import numpy as np

CLASS_VALIDATED_CORRELATION = {
    # "urban": 0.00,        # <- fill from held-out, per-classifier-class results
    # "vegetation": 0.00,
    # "bare_terrain": 0.00,
    # "water": 0.00,
    # "shadow": 0.00,
    # "unknown_low_confidence": 0.00,
}
DEFAULT_VALIDATION = 0.3
_warned = set()


def _sample_confidence(n, min_pixels, max_reliable_pixels):
    if n <= 0:
        return 0.0
    if n < min_pixels:
        return 0.3 * n / min_pixels
    if n >= max_reliable_pixels:
        return 1.0
    return 0.3 + 0.7 * (n - min_pixels) / (max_reliable_pixels - min_pixels)


def compute_confidence(terrain_mask, fit_report=None, class_names=None,
                       min_pixels=500, max_reliable_pixels=5000):
    """fit_report is accepted for backward compatibility but not used."""
    conf = np.zeros(terrain_mask.shape, dtype=np.float32)
    for class_id in np.unique(terrain_mask):
        in_class = terrain_mask == class_id
        name = (class_names or {}).get(int(class_id), f"class_{int(class_id)}")
        validation = CLASS_VALIDATED_CORRELATION.get(name)
        if validation is None:
            validation = DEFAULT_VALIDATION
            if name not in _warned:
                print(f"[confidence] no validated correlation for class '{name}' "
                      f"- using placeholder {DEFAULT_VALIDATION}")
                _warned.add(name)
        conf[in_class] = _sample_confidence(
            int(in_class.sum()), min_pixels, max_reliable_pixels) * validation
    return conf