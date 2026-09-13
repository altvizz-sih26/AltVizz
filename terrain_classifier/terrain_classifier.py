"""
DepthWizard - Terrain / Semantic Classification (MVP)
------------------------------------------------------
Lightweight, explainable OpenCV heuristic classifier.
NO deep learning. NO training. Pure color / brightness / texture / shape rules.

Output per image:
    terrain_mask.npy          -> uint8 array, same H,W as depth.npy, values in {0..5}
    terrain_mask_preview.png  -> BGR color-coded visualization

Class IDs:
    0 = urban
    1 = vegetation
    2 = bare_terrain
    3 = water
    4 = shadow
    5 = unknown_low_confidence  (fallback - pixel didn't clearly match anything)

Author: <your name>, SIH 2026 - DepthWizard
"""

import cv2
import numpy as np
import os


# ============================================================================
# 1. CONFIG - all tunable thresholds live here so you never have to hunt
#    through the logic to change a number. Tune these against your GAMUS
#    samples before the demo.
# ============================================================================

CONFIG = {
    # ---- class IDs (do not change unless you also update the preview colors) ----
    "CLASS_URBAN": 0,
    "CLASS_VEGETATION": 1,
    "CLASS_BARE_TERRAIN": 2,
    "CLASS_WATER": 3,
    "CLASS_SHADOW": 4,
    "CLASS_UNKNOWN": 5,

    # ---- vegetation (HSV) ----
    # Hue is measured 0-179 in OpenCV. Green foliage typically falls ~35-85.
    "VEG_HUE_LOW": 30,
    "VEG_HUE_HIGH": 85,          # trimmed slightly (was 90) so it doesn't bleed into water's cyan range
    "VEG_SAT_LOW": 30,           # reject washed-out/near-gray pixels
    "VEG_VAL_LOW": 25,           # reject near-black pixels (those go to shadow instead)
    "VEG_MIN_AREA": 40,          # px; drop connected components smaller than this (noise)
    "VEG_MORPH_KERNEL": 3,

    # ---- water (HSV hue + smoothness) ----
    # WHY: water is the new class. Two independent cues are combined:
    #   (a) hue sits in the blue/cyan band
    #   (b) surface is locally SMOOTH (low local variance) - water rarely
    #       has the fine texture that blue rooftops/tarps/vehicles do.
    "WATER_HUE_LOW": 85,
    "WATER_HUE_HIGH": 135,
    "WATER_SAT_LOW": 15,         # water can be fairly desaturated (murky/overcast) - keep loose
    "WATER_VAL_LOW": 20,         # exclude near-black (that's shadow's job)
    "WATER_LOCAL_STD_MAX": 12.0, # local grayscale std-dev threshold; below = "smooth enough" to be water
    "WATER_TEXTURE_WINDOW": 7,   # window size (px) for local smoothness check
    "WATER_MIN_AREA": 60,
    "WATER_MORPH_KERNEL": 3,

    # ---- shadow (HSV-V + LAB-L) ----
    "SHADOW_V_MAX": 60,          # HSV Value below this = dark
    "SHADOW_L_MAX": 70,          # LAB L below this = dark (0-255 scale)
    "SHADOW_MIN_AREA": 30,

    # ---- urban (edges/contours - was 'building') ----
    "CANNY_LOW": 50,
    "CANNY_HIGH": 150,
    "URBAN_DILATE_KERNEL": 3,
    "URBAN_DILATE_ITERS": 2,
    "URBAN_MIN_CONTOUR_AREA": 150,        # px; ignore tiny contours (noise, not a structure)
    "URBAN_MAX_CONTOUR_AREA_FRAC": 0.5,   # ignore contours covering >50% of image
    "URBAN_MIN_EXTENT": 0.55,             # contour_area / bbox_area -> "boxiness"
    "URBAN_MIN_ASPECT": 0.2,              # reject very thin slivers (roads/cracks)
    "URBAN_MAX_ASPECT": 5.0,
    "URBAN_MORPH_KERNEL": 5,

    # ---- bare terrain (explicit heuristic, NOT pure leftover) ----
    # WHY explicit instead of "whatever's left": this lets genuinely
    # ambiguous pixels fall through to 'unknown' instead of being forced
    # into bare_terrain just because nothing else claimed them.
    "BARE_SAT_MAX": 55,           # soil/rock/dry ground is fairly desaturated
    "BARE_VAL_LOW": 60,           # not too dark (that's shadow)
    "BARE_VAL_HIGH": 230,         # not blown-out highlight
    "BARE_LOCAL_STD_MAX": 25.0,   # relatively low texture (not sharp urban edges)
    "BARE_TEXTURE_WINDOW": 7,

    # ---- misc ----
    "GAUSSIAN_BLUR_KSIZE": 3,     # light denoise before all heuristics
}


# ============================================================================
# 2. PREVIEW COLOR MAP (BGR, since we use OpenCV to save the PNG)
# ============================================================================

PREVIEW_COLORS_BGR = {
    CONFIG["CLASS_UNKNOWN"]:       (128, 128, 128),  # gray
    CONFIG["CLASS_VEGETATION"]:    (0, 180, 0),       # green
    CONFIG["CLASS_URBAN"]:         (0, 0, 220),       # red
    CONFIG["CLASS_BARE_TERRAIN"]:  (0, 200, 220),     # tan/yellow-ish
    CONFIG["CLASS_WATER"]:         (220, 120, 0),     # blue
    CONFIG["CLASS_SHADOW"]:        (160, 32, 32),     # dark navy
}


# ============================================================================
# 3. SHARED HELPERS
# ============================================================================

def _remove_small_components(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    """
    WHY: shared connected-component filter. Any class mask with scattered
    single-pixel or few-pixel blobs looks unprofessional and is almost
    certainly noise, not a real object.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    cleaned = np.zeros_like(mask_u8)
    for label_id in range(1, num_labels):  # skip label 0 = background
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = 255
    return cleaned


def _local_std(gray: np.ndarray, window: int) -> np.ndarray:
    """
    WHY: a fast local-texture estimate using box filters instead of a
    per-pixel Python loop. std = sqrt(E[x^2] - (E[x])^2) computed over a
    sliding window via cv2.blur (which is a fast box filter).
    Used to tell 'smooth' surfaces (water, flat bare ground) apart from
    'textured' surfaces (urban edges, vegetation clutter).
    """
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (window, window))
    sq_mean = cv2.blur(gray_f * gray_f, (window, window))
    variance = np.clip(sq_mean - mean * mean, 0, None)
    return np.sqrt(variance)


# ============================================================================
# 4. INDIVIDUAL HEURISTICS
#    Each returns a boolean mask (H, W) - True where that class is detected.
# ============================================================================

def detect_vegetation(hsv: np.ndarray, cfg: dict) -> np.ndarray:
    """
    WHY: Healthy vegetation reflects strongly in the green band and has
    reasonable saturation (unlike gray concrete/roads which are desaturated).
    """
    lower = np.array([cfg["VEG_HUE_LOW"], cfg["VEG_SAT_LOW"], cfg["VEG_VAL_LOW"]])
    upper = np.array([cfg["VEG_HUE_HIGH"], 255, 255])
    raw_mask = cv2.inRange(hsv, lower, upper)

    k = cfg["VEG_MORPH_KERNEL"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
    raw_mask = _remove_small_components(raw_mask, cfg["VEG_MIN_AREA"])

    return raw_mask > 0


def detect_water(hsv: np.ndarray, gray: np.ndarray, exclude_mask: np.ndarray, cfg: dict) -> np.ndarray:
    """
    WHY two cues combined: hue alone would confuse water with any blue
    object (tarps, blue rooftops, pools painted vividly). Smoothness alone
    would confuse water with flat bare ground or shadowed flat surfaces.
    Requiring BOTH color-in-blue-band AND local smoothness is a much
    stronger (though still imperfect) signal specifically for water.
    """
    lower = np.array([cfg["WATER_HUE_LOW"], cfg["WATER_SAT_LOW"], cfg["WATER_VAL_LOW"]])
    upper = np.array([cfg["WATER_HUE_HIGH"], 255, 255])
    hue_mask = cv2.inRange(hsv, lower, upper) > 0

    local_std = _local_std(gray, cfg["WATER_TEXTURE_WINDOW"])
    smooth_mask = local_std < cfg["WATER_LOCAL_STD_MAX"]

    water_mask = hue_mask & smooth_mask & (~exclude_mask)

    water_u8 = (water_mask.astype(np.uint8)) * 255
    k = cfg["WATER_MORPH_KERNEL"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    water_u8 = cv2.morphologyEx(water_u8, cv2.MORPH_CLOSE, kernel)
    water_u8 = _remove_small_components(water_u8, cfg["WATER_MIN_AREA"])

    return water_u8 > 0


def detect_shadow(hsv: np.ndarray, lab: np.ndarray, exclude_mask: np.ndarray, cfg: dict) -> np.ndarray:
    """
    WHY: shadows are dark in both HSV-V and LAB-L. Requiring BOTH reduces
    false positives from moderately-dark-but-not-shadowed surfaces.
    WHY exclude_mask: dark vegetation/water should not be re-labeled shadow -
    those are evaluated in earlier, higher-priority stages.
    """
    v_channel = hsv[:, :, 2]
    l_channel = lab[:, :, 0]

    dark_v = v_channel < cfg["SHADOW_V_MAX"]
    dark_l = l_channel < cfg["SHADOW_L_MAX"]

    shadow_mask = dark_v & dark_l & (~exclude_mask)

    shadow_u8 = (shadow_mask.astype(np.uint8)) * 255
    shadow_u8 = _remove_small_components(shadow_u8, cfg["SHADOW_MIN_AREA"])

    return shadow_u8 > 0


def detect_urban(gray: np.ndarray, exclude_mask: np.ndarray, cfg: dict) -> np.ndarray:
    """
    WHY edges+contours instead of color: buildings/roads/urban structures
    have no consistent color but DO tend to have strong edges and regular
    outlines. This is the weakest heuristic here - it over-triggers on
    parking lots, containers, concrete slabs, and under-triggers on small
    or irregular rooftops. Flag this explicitly in the viva.
    """
    edges = cv2.Canny(gray, cfg["CANNY_LOW"], cfg["CANNY_HIGH"])

    k = cfg["URBAN_DILATE_KERNEL"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    edges_dilated = cv2.dilate(edges, kernel, iterations=cfg["URBAN_DILATE_ITERS"])

    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    total_area = h * w
    urban_mask = np.zeros((h, w), dtype=np.uint8)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg["URBAN_MIN_CONTOUR_AREA"]:
            continue
        if area > total_area * cfg["URBAN_MAX_CONTOUR_AREA_FRAC"]:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        bbox_area = bw * bh
        if bbox_area == 0:
            continue

        extent = area / bbox_area
        if extent < cfg["URBAN_MIN_EXTENT"]:
            continue

        aspect_ratio = bw / float(bh)
        if not (cfg["URBAN_MIN_ASPECT"] <= aspect_ratio <= cfg["URBAN_MAX_ASPECT"]):
            continue

        cv2.drawContours(urban_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    k2 = cfg["URBAN_MORPH_KERNEL"]
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    urban_mask = cv2.morphologyEx(urban_mask, cv2.MORPH_CLOSE, kernel2)

    urban_bool = (urban_mask > 0) & (~exclude_mask)
    return urban_bool


def detect_bare_terrain(hsv: np.ndarray, gray: np.ndarray, exclude_mask: np.ndarray, cfg: dict) -> np.ndarray:
    """
    WHY this is an EXPLICIT positive check rather than 'whatever is left':
    forcing every unassigned pixel into bare_terrain would hide genuinely
    ambiguous pixels. Instead, bare_terrain requires: low saturation
    (soil/rock/dry ground isn't vivid), a mid brightness range (not shadow,
    not blown-out highlight), and low-to-moderate local texture (not the
    sharp edges typical of urban structures). Anything that fails this
    positive test falls through to 'unknown / low_confidence' instead.
    """
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    low_sat = s_channel < cfg["BARE_SAT_MAX"]
    mid_val = (v_channel > cfg["BARE_VAL_LOW"]) & (v_channel < cfg["BARE_VAL_HIGH"])

    local_std = _local_std(gray, cfg["BARE_TEXTURE_WINDOW"])
    low_texture = local_std < cfg["BARE_LOCAL_STD_MAX"]

    bare_mask = low_sat & mid_val & low_texture & (~exclude_mask)
    return bare_mask


# ============================================================================
# 5. MAIN CLASSIFICATION FUNCTION - applies priority order
# ============================================================================

def classify_terrain(image_bgr: np.ndarray, cfg: dict = CONFIG) -> np.ndarray:
    """
    Runs all heuristics and combines them using a FIXED PRIORITY ORDER:
        1. vegetation    (color - most trustworthy)
        2. water         (color + smoothness - fairly trustworthy)
        3. shadow        (brightness - evaluated on non-veg/water only)
        4. urban         (shape - evaluated on non-veg/water/shadow only)
        5. bare_terrain  (positive color+texture check on what's left)
        6. unknown       (whatever still doesn't fit anything - genuine
                          low-confidence residual, NOT a forced bucket)

    Returns: uint8 array (H, W) of class IDs. Exact same H,W as input image,
    which you must guarantee is exact same H,W as depth.npy upstream.
    """
    k = cfg["GAUSSIAN_BLUR_KSIZE"]
    denoised = cv2.GaussianBlur(image_bgr, (k, k), 0)

    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    mask = np.full((h, w), cfg["CLASS_UNKNOWN"], dtype=np.uint8)
    assigned = np.zeros((h, w), dtype=bool)

    # --- Stage 1: vegetation ---
    veg_mask = detect_vegetation(hsv, cfg)
    mask[veg_mask] = cfg["CLASS_VEGETATION"]
    assigned |= veg_mask

    # --- Stage 2: water (only where not vegetation) ---
    water_mask = detect_water(hsv, gray, assigned, cfg)
    mask[water_mask] = cfg["CLASS_WATER"]
    assigned |= water_mask

    # --- Stage 3: shadow (only where not vegetation/water) ---
    shadow_mask = detect_shadow(hsv, lab, assigned, cfg)
    mask[shadow_mask] = cfg["CLASS_SHADOW"]
    assigned |= shadow_mask

    # --- Stage 4: urban (only where not vegetation/water/shadow) ---
    urban_mask = detect_urban(gray, assigned, cfg)
    mask[urban_mask] = cfg["CLASS_URBAN"]
    assigned |= urban_mask

    # --- Stage 5: bare terrain (positive check on remaining pixels) ---
    bare_mask = detect_bare_terrain(hsv, gray, assigned, cfg)
    mask[bare_mask] = cfg["CLASS_BARE_TERRAIN"]
    assigned |= bare_mask

    # --- Stage 6: unknown / low_confidence = whatever is still unassigned ---
    # (mask is already initialized to CLASS_UNKNOWN, so no action needed -
    # this line is just here to make the final state explicit/readable)
    mask[~assigned] = cfg["CLASS_UNKNOWN"]

    return mask


# ============================================================================
# 6. PREVIEW GENERATION
# ============================================================================

def make_preview(mask: np.ndarray, color_map: dict = PREVIEW_COLORS_BGR) -> np.ndarray:
    """
    Converts integer class-ID mask into a BGR color image for visual QA
    and demo slides. Kept fully separate from terrain_mask.npy so the
    .npy always stays pure integer IDs.
    """
    h, w = mask.shape
    preview = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        preview[mask == class_id] = color
    return preview


# ============================================================================
# 7. I/O HELPERS
# ============================================================================

def save_outputs(mask: np.ndarray, preview_bgr: np.ndarray, out_dir: str, prefix: str = ""):
    """
    WHY prefix: without it, every image would overwrite the same
    'terrain_mask.npy' / 'terrain_mask_preview.png' in out_dir. Passing the
    source image's own base filename as prefix (e.g. 'img2') produces
    'img2_terrain_mask.npy' / 'img2_terrain_mask_preview.png', so outputs
    for multiple images can safely live in the same folder and you can
    tell at a glance which mask belongs to which image.
    """
    os.makedirs(out_dir, exist_ok=True)

    # WHY the underscore only when prefix is non-empty: keeps the plain
    # 'terrain_mask.npy' name available for callers who don't pass a prefix
    # (e.g. quick one-off tests), instead of producing '_terrain_mask.npy'.
    mask_name = f"{prefix}_terrain_mask.npy" if prefix else "terrain_mask.npy"
    preview_name = f"{prefix}_terrain_mask_preview.png" if prefix else "terrain_mask_preview.png"

    mask_path = os.path.join(out_dir, mask_name)
    preview_path = os.path.join(out_dir, preview_name)

    np.save(mask_path, mask)
    cv2.imwrite(preview_path, preview_bgr)

    return mask_path, preview_path


def run_on_image(image_path: str, out_dir: str, depth_path: str = None, cfg: dict = CONFIG):
    """
    Full pipeline for ONE image: load -> classify -> preview -> save -> verify.
    If depth_path is given, verifies shape alignment against depth.npy.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # WHY derive prefix from image_path here: this guarantees the saved
    # filenames always match the INPUT image's name (e.g. 'img2.png' ->
    # 'img2_terrain_mask.npy'), the same convention run.py already uses
    # for depth.npy, so the two are trivially pairable by filename.
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    mask = classify_terrain(image_bgr, cfg)
    preview = make_preview(mask, PREVIEW_COLORS_BGR)
    mask_path, preview_path = save_outputs(mask, preview, out_dir, prefix=base_name)

    print(f"[OK] Image shape (H,W,C): {image_bgr.shape}")
    print(f"[OK] Mask shape  (H,W)  : {mask.shape}")
    print(f"[OK] Unique class IDs   : {np.unique(mask).tolist()}")
    print(f"[OK] Saved: {mask_path}")
    print(f"[OK] Saved: {preview_path}")

    if depth_path is not None:
        depth = np.load(depth_path)
        if depth.shape[:2] != mask.shape:
            raise ValueError(
                f"SHAPE MISMATCH: depth.npy is {depth.shape[:2]} but "
                f"terrain_mask.npy is {mask.shape}. Do NOT proceed to fusion "
                f"until these match exactly."
            )
        print(f"[OK] depth.npy shape    : {depth.shape} -> matches mask spatial dims.")

    return mask, preview


# ============================================================================
# 8. CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DepthWizard terrain classifier (heuristic MVP)")
    parser.add_argument("--image", required=True, help="Path to input RGB image")
    parser.add_argument("--depth", required=False, default=None, help="Path to depth.npy (optional, for shape verification)")
    parser.add_argument("--out", required=False, default="./terrain_output", help="Output directory")
    args = parser.parse_args()

    run_on_image(args.image, args.out, args.depth)
