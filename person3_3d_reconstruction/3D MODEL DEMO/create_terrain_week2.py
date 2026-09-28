"""
Automatic 3D Terrain / DSM Mesh Generator
Person 3 - 3D Reconstruction

INPUT:
    BEFORE RGB + BEFORE elevation
    DURING RGB + DURING elevation

OUTPUT:
    before.glb
    during.glb

GOAL (what this version fixes):
    - Flat ground stays FLAT (no "carpet fold" noise/waviness)
    - Buildings rise with a gentle UPLIFTED look -- not needle-thin
      spikes. Isolated 1-2 pixel noise no longer produces tall thin
      "poles"; real building footprints get a soft, rounded-off top
      instead of a razor-sharp point.
    - Mountains keep their natural continuous shape (no random patchy
      gaps caused by naive downsampling)
    - Everything is computed automatically FROM THE DATA -- no numbers
      are hardcoded for one specific dataset, so any before/during
      image+elevation pair a backend user uploads works the same way.
    - NODATA (-9999) elevation sentinel pixels no longer leak through
      as extreme spikes ("foggy"/jagged patches in the rendered mesh).
    - NEW -- NODATA / invalid pixels are now filled using NEAREST-VALID-
      NEIGHBOR interpolation instead of one single global median value.
      A single flat fill was turning entire mountains "flat" whenever a
      DSM had data gaps on steep slopes (very common in real datasets),
      leaving only a few surviving genuine points sticking up as spikes.
      Nearest-fill keeps the local slope continuous, so the mountain's
      real shape survives and reads as one smooth uplifted mass instead
      of "mostly flat + isolated spikes".
    - Tile-edge "collapsed cliff" look at the mesh's outer border is
      removed by auto-trimming a small margin off the whole grid
      (elevation AND texture together) before meshing.

HOW (technique):
    1. Elevation is split into two layers:
         base   = large-neighborhood MEDIAN of elevation
                  -> keeps mountains' true shape, removes small objects
                     like buildings from this layer
         detail = elevation - base
                  -> buildings/small bumps show up here as spikes;
                     everywhere else this is just sensor noise
    2. detail is passed through an automatic noise threshold (MAD --
       median absolute deviation): anything below it is zeroed (flat
       ground stays flat), anything above it is kept.
    3. spike cleanup on the kept detail:
         a. morphological "opening" removes any kept region smaller
            than a couple of pixels across -- these are noise spikes,
            not real buildings (a real building covers a solid patch
            of pixels, not a single dot)
         b. a small gaussian smoothing pass softens the remaining
            peaks so buildings read as gently "uplifted" blocks rather
            than sharp needle points
    4. base + cleaned_detail = final elevation used for Z.
    5. Downsampling for mesh resolution uses AREA-AVERAGE (block mean),
       never skips real terrain data.
    6. Vertical exaggeration ("relief scale") is computed automatically
       from each dataset's own elevation spread (percentile-based).
    7. A thin border margin (percentage of the grid) is trimmed from
       elevation AND texture together before meshing.

CHANGELOG (this version):
    - load_elevation() now fills invalid/NODATA pixels using nearest-
      valid-neighbor interpolation (distance_transform_edt) instead of
      a single global median -- fixes mountains reading as "flat +
      isolated spikes" when the source DSM has data gaps on slopes.
    - auto_trim_border() removes a percentage-based margin from the
      whole grid (elevation + texture, kept in sync).
    - spike cleanup (morphological opening + light smoothing) on the
      buildings/detail layer so features read as "uplifted" rather
      than thin spiky poles.
    - Default relief_target_pct lowered slightly (0.30 -> 0.20) for a
      gentler overall exaggeration; raise it back via
      --relief-target-pct if terrain still looks too flat.
"""

import os
import argparse

import numpy as np
import trimesh

from PIL import Image
from scipy.ndimage import (
    median_filter, gaussian_filter, binary_opening, binary_dilation, zoom,
    distance_transform_edt, label, find_objects,
)

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


# ============================================================
# CONSTANTS
# ============================================================

NODATA_VALUE = -9999.0  # project-wide sentinel for "no real elevation data here"

DEFAULT_BORDER_TRIM_PCT = 0.05  # default 5% margin trimmed from each edge
                                  # of the WHOLE grid (elevation + texture)
                                  # to remove DSM tile-edge artifacts


# ============================================================
# 1. LOAD ELEVATION
# ============================================================

def load_elevation(source):
    """Accepts a .npy file path OR an already-loaded numpy array."""

    if isinstance(source, np.ndarray):
        elevation = source.copy()
    else:
        elevation = np.load(source)

    elevation = np.asarray(elevation, dtype=np.float32)

    if elevation.ndim != 2:
        raise ValueError(f"Elevation must be 2D. Got {elevation.shape}")

    # Explicitly exclude the -9999 nodata sentinel, not just NaN/Inf.
    valid = np.isfinite(elevation) & (elevation != NODATA_VALUE)

    if not np.any(valid):
        raise ValueError("Elevation contains no valid values.")

    if not np.all(valid):
        # Nearest-valid-neighbor fill instead of one flat global value.
        #
        # A single flat fill (old behavior: elevation[~valid] = median)
        # turns real terrain -- mountains especially, which commonly
        # have DSM data gaps on steep slopes from stereo-matching
        # occlusion -- into "mostly flat ground + a few leftover
        # spikes" (the surviving genuine points stick up above the
        # artificial flat floor). Nearest-fill instead propagates each
        # hole's value from its closest valid pixel, so a hole inside
        # a mountain slope gets a mountain-like value and the slope
        # stays geometrically continuous.
        _, indices = distance_transform_edt(~valid, return_indices=True)
        elevation = elevation[tuple(indices)]

    return elevation


def align_shapes(before: np.ndarray, during: np.ndarray):
    """Auto-resample DURING onto BEFORE's grid if resolutions differ."""
    if before.shape == during.shape:
        return before, during

    print(f"[info] elevation shapes differ {before.shape} vs {during.shape} "
          f"-> auto-resampling DURING onto BEFORE's grid")

    zy = before.shape[0] / during.shape[0]
    zx = before.shape[1] / during.shape[1]
    during_aligned = zoom(during, (zy, zx), order=1)

    during_aligned = during_aligned[:before.shape[0], :before.shape[1]]
    if during_aligned.shape != before.shape:
        pad_h = before.shape[0] - during_aligned.shape[0]
        pad_w = before.shape[1] - during_aligned.shape[1]
        during_aligned = np.pad(during_aligned, ((0, max(0, pad_h)), (0, max(0, pad_w))), mode="edge")
        during_aligned = during_aligned[:before.shape[0], :before.shape[1]]

    return before, during_aligned


def auto_trim_border(array: np.ndarray, pct: float):
    """
    Trims a percentage-based margin off ALL FOUR edges of a 2D (elevation)
    or 3D (H, W, C texture) array.
    """
    if pct <= 0:
        return array

    h, w = array.shape[0], array.shape[1]
    trim_h = int(round(h * pct))
    trim_w = int(round(w * pct))

    if trim_h <= 0 and trim_w <= 0:
        return array

    trim_h = min(trim_h, (h - 1) // 2)
    trim_w = min(trim_w, (w - 1) // 2)

    if array.ndim == 2:
        return array[trim_h:h - trim_h, trim_w:w - trim_w]
    else:
        return array[trim_h:h - trim_h, trim_w:w - trim_w, :]


def clip_elevation_outliers(elevation: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Clips extreme values (DEM border noise) using percentiles computed from the data itself."""
    lo = np.percentile(elevation, low_pct)
    hi = np.percentile(elevation, high_pct)
    return np.clip(elevation, lo, hi)


def remove_planar_tilt(elevation: np.ndarray, iterations: int = 4, keep_pct: float = 60.0) -> np.ndarray:
    """
    Removes a large-scale systematic tilt/slope across the whole image
    (a photogrammetry/DSM reconstruction artifact). Fit iteratively
    using only the lower ("ground-like") points so mountains/buildings
    can't skew the fit.
    """
    H, W = elevation.shape
    rows, cols = np.mgrid[0:H, 0:W]

    X = cols.ravel().astype(np.float64)
    Y = rows.ravel().astype(np.float64)
    Z = elevation.ravel().astype(np.float64)

    mask = np.ones_like(Z, dtype=bool)
    coeffs = None

    for _ in range(iterations):
        A = np.column_stack([X[mask], Y[mask], np.ones(mask.sum())])
        coeffs, _, _, _ = np.linalg.lstsq(A, Z[mask], rcond=None)

        trend_full = coeffs[0] * X + coeffs[1] * Y + coeffs[2]
        residual = Z - trend_full

        threshold = np.percentile(residual, keep_pct)
        mask = residual <= threshold

    trend = (coeffs[0] * X + coeffs[1] * Y + coeffs[2]).reshape(H, W)
    detrended = elevation - trend

    print(f"  removed planar tilt (slope_x={coeffs[0]:.5f}, slope_y={coeffs[1]:.5f} per pixel, "
          f"fit using bottom {keep_pct:.0f}% ground-like points)")

    return detrended.astype(np.float32)


# ============================================================
# 2. GROUND-FLATTEN + SHARP-OBJECT SEPARATION
# ============================================================

def _auto_base_kernel_size(height: int, width: int) -> int:
    """Neighborhood size for the 'base terrain' median filter, scaled to image resolution."""
    size = max(9, min(height, width) // 35)
    if size % 2 == 0:
        size += 1
    return size


def separate_relief(elevation: np.ndarray, spike_smooth_sigma: float = 1.2,
                     min_feature_px: int = None, mad_multiplier: float = 5.0,
                     mountain_detail_boost: float = 1.3,
                     large_feature_multiplier: float = 3.0):
    """
    Splits elevation into:
      base           -> mountains/ground large-scale shape (median filter)
      cleaned_detail -> buildings/local features ONLY

    Ground stays genuinely flat because:
      0. A tiny pre-denoise pass (median size=3) removes single-pixel
         sensor/matching noise BEFORE anything else -- this is the
         noise that was previously showing up as spikes across the
         ENTIRE surface (grass, roads, everywhere), not just buildings.
      1. The noise-vs-feature threshold is conservative
         (`mad_multiplier`, default 5x MAD) so only pixels that clearly
         stand out get flagged as a real feature.
      2. `min_feature_px` (auto-scaled from the base kernel size, i.e.
         proportional to how "large-scale" the terrain layer already
         is -- not a fixed number) requires a solid contiguous patch of
         pixels, not a lone dot, to count as a real building.
      3. SIZE-ADAPTIVE SMOOTHING (fixes "flat plateau + spike" mountains):
         a single small `spike_smooth_sigma` works for a real building
         (a few pixels wide) but leaves a big real terrain feature (a
         hillside, a plateau a whole area sits on) with a hard, cliff-
         like edge, because that edge also survives the opening filter
         as one large contiguous patch. Each contiguous kept region is
         labeled and measured on its own: small regions keep the crisp
         `spike_smooth_sigma`; any region whose extent is at least
         `large_feature_multiplier` times the base kernel size (i.e. a
         genuine terrain-scale feature, not a building) is smoothed with
         a much larger sigma, proportional to its own size, so its edges
         blend into the surrounding terrain like a real slope instead of
         a vertical wall.
    """
    H, W = elevation.shape
    base_kernel = _auto_base_kernel_size(H, W)

    if min_feature_px is None:
        min_feature_px = max(3, base_kernel // 5)

    # 0. tiny pre-denoise: removes single-pixel sensor noise so it can
    #    never masquerade as a "feature" later on.
    denoised = median_filter(elevation, size=3, mode="nearest")

    # mode="nearest" avoids scipy's default mirrored-edge artifact.
    base = median_filter(denoised, size=base_kernel, mode="nearest")

    # Establish the ground-floor reference from the STABLE (pre-boost)
    # base -- this must be computed before any amplification, otherwise
    # a boost-created dip could corrupt its own reference and survive.
    ground_floor = np.percentile(base, 5)

    # Enhance mountain/ridge definition: split base into an overall
    # silhouette (very broadly smoothed) and the mid-scale ridge/slope
    # detail sitting on top of it, then amplify that ridge detail a bit.
    # This makes mountain shapes read clearly as a mountain (visible
    # ridges/slopes) instead of one soft rounded blob, WITHOUT touching
    # the buildings layer at all.
    broad_silhouette = gaussian_filter(base, sigma=base_kernel)
    ridge_detail = base - broad_silhouette

    # Clip ridge_detail to a robust percentile range BEFORE boosting --
    # this is what previously went wrong: a handful of leftover
    # single-pixel outliers got multiplied into towering needle spikes.
    # Clipping first means only genuine, widespread terrain variation
    # gets amplified, not rare outliers.
    r_lo, r_hi = np.percentile(ridge_detail, [2, 98])
    ridge_detail = np.clip(ridge_detail, r_lo, r_hi)

    base = broad_silhouette + ridge_detail * mountain_detail_boost

    # Flatten any dip BELOW the (pre-boost) ground floor -- guaranteed
    # to catch any dip even if boosting created a new one, since this
    # reference was captured before boosting could corrupt it.
    base = np.clip(base, ground_floor, None)

    detail = denoised - base

    # Spatially-ADAPTIVE noise threshold (fixes flat-mountain-with-a-few-
    # spikes): a single global MAD number assumes noise looks the same
    # everywhere in the image. Real DSMs don't work that way -- smooth
    # areas (flat city ground) are naturally low-noise, while rough
    # natural terrain (mountains, vegetation, slopes) is naturally
    # noisier from photogrammetry matching error. A global threshold
    # tuned for the smooth area is too LOW for the naturally-rougher
    # mountain area, so most of its real, gentle terrain roughness gets
    # misread as "features" -- some of it flattened away as sub-
    # threshold noise, some of it surviving as isolated spikes.
    #
    # Instead, compute the noise floor LOCALLY: for every pixel, compare
    # it against the typical noise level of its OWN neighborhood, not
    # the whole image. The neighborhood is sized well beyond a single
    # building footprint (a multiple of the base kernel) so it reflects
    # general local terrain roughness, not individual small features.
    local_stat_kernel = base_kernel * 3
    if local_stat_kernel % 2 == 0:
        local_stat_kernel += 1

    local_median = median_filter(detail, size=local_stat_kernel, mode="nearest")
    local_mad = median_filter(np.abs(detail - local_median), size=local_stat_kernel, mode="nearest")

    # Global MAD still sets a sane floor, so a perfectly flat, noise-free
    # region can't end up with a near-zero threshold that flags every
    # last sub-pixel wobble as a "feature".
    global_mad = np.median(np.abs(detail - np.median(detail)))
    global_floor = global_mad if global_mad > 1e-6 else np.std(detail) * 1.0

    noise_threshold = mad_multiplier * np.maximum(local_mad, global_floor * 0.5)

    kept_mask = np.abs(detail) > noise_threshold

    # Require a solid patch of pixels (real building footprint), not an
    # isolated dot, to survive.
    structure = np.ones((min_feature_px, min_feature_px))
    kept_mask = binary_opening(kept_mask, structure=structure)

    # Widen (dilate) each surviving feature's footprint a couple of
    # pixels. Fixes "needle spikes": gaussian-smoothing only softens a
    # feature's PEAK, it doesn't widen its BASE. A feature that is only
    # 3-4px wide still looks like a thin pole once the vertical
    # exaggeration scale is applied, no matter how soft its top is. A
    # small dilation gives every real feature an actual footprint width,
    # so after vertical scaling it reads as a short raised block instead
    # of a tall thin spike.
    dilate_px = max(1, min_feature_px // 2)
    kept_mask = binary_dilation(kept_mask, structure=np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1)))

    # Only allow UPWARD features (buildings, mountains rising above
    # ground) -- never downward. Any negative detail was creating
    # "holes"/pits in roads and flat ground, which isn't a real feature,
    # just noise on the low side. Clipping to >=0 removes every hole
    # while leaving every real uplifted feature untouched.
    detail = np.clip(detail, 0, None)
    kept_mask = kept_mask & (detail > 0)

    # Cap any single feature's height using the data's OWN distribution
    # of kept feature heights (percentile-based, never a hardcoded
    # number). Without this, one unusually tall outlier cluster (a
    # single bad pixel, a rare tall real structure) sticks up far above
    # every other feature once vertical exaggeration is applied, reading
    # as one dramatic "super spike" against otherwise modest bumps.
    kept_values = detail[kept_mask]
    if kept_values.size > 0:
        height_cap = np.percentile(kept_values, 95)
        if height_cap > 0:
            detail = np.minimum(detail, height_cap)

    # ---- size-adaptive smoothing, per contiguous feature ----
    # A single global sigma can't serve both a 4px building AND a
    # 200px hillside/plateau at once. Label every surviving contiguous
    # region separately and size each one's smoothing individually,
    # working only inside a small padded window around that region
    # (so this stays fast even with hundreds of small building labels).
    large_feature_min_extent = base_kernel * large_feature_multiplier

    cleaned_detail = np.zeros_like(detail)
    labeled, num_features = label(kept_mask)

    if num_features > 0:
        objects = find_objects(labeled)

        for comp_id, sl in enumerate(objects, start=1):
            if sl is None:
                continue

            r0, r1 = sl[0].start, sl[0].stop
            c0, c1 = sl[1].start, sl[1].stop
            comp_extent = max(r1 - r0, c1 - c0)

            if comp_extent >= large_feature_min_extent:
                # Genuine terrain-scale feature (mountain shoulder,
                # plateau, hillside) -- smooth proportional to its own
                # size so the edge reads as a slope, not a cliff.
                sigma = max(spike_smooth_sigma, comp_extent / 8.0)
            else:
                # Real building-scale feature -- keep the original,
                # crisp "uplifted block" look.
                sigma = spike_smooth_sigma

            pad = int(sigma * 4) + 2
            pr0, pr1 = max(0, r0 - pad), min(H, r1 + pad)
            pc0, pc1 = max(0, c0 - pad), min(W, c1 + pad)

            local_mask = labeled[pr0:pr1, pc0:pc1] == comp_id
            local_detail = np.where(local_mask, detail[pr0:pr1, pc0:pc1], 0.0)

            if sigma > 0:
                local_detail = gaussian_filter(local_detail, sigma=sigma)

            cleaned_detail[pr0:pr1, pc0:pc1] += local_detail

    cleaned_detail = np.clip(cleaned_detail, 0, None)

    return (base + cleaned_detail).astype(np.float32)


def block_mean_downsample(elevation: np.ndarray, step: int) -> np.ndarray:
    """Area-average downsampling (every step x step block is averaged) -- never skips data."""
    if step <= 1:
        return elevation

    H, W = elevation.shape
    newH = H - (H % step)
    newW = W - (W % step)
    cropped = elevation[:newH, :newW]
    reshaped = cropped.reshape(newH // step, step, newW // step, step)
    return reshaped.mean(axis=(1, 3)).astype(np.float32)


# ============================================================
# 3. LOAD RGB / TIFF  (jpg / jpeg / png / tif / tiff, any resolution)
# ============================================================

def load_texture(source):
    """Accepts a file path (.jpg/.jpeg/.png/.tif/.tiff) OR a numpy array."""

    if isinstance(source, np.ndarray):
        image = source
    else:
        extension = os.path.splitext(source)[1].lower()

        if extension in [".tif", ".tiff"]:
            image = _load_tif_any(source)
        elif extension in [".jpg", ".jpeg", ".png"]:
            image = np.array(Image.open(source))
        else:
            raise ValueError("Supported formats: TIF, TIFF, JPG, JPEG, PNG")

    image = np.asarray(image)

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RGB image expected. Got {image.shape}")

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        min_value = np.nanmin(image)
        max_value = np.nanmax(image)

        if max_value > min_value:
            image = (image - min_value) / (max_value - min_value) * 255.0
        else:
            image = np.zeros_like(image)

        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def _load_tif_any(path: str) -> np.ndarray:
    try:
        return np.array(Image.open(path).convert("RGB"))
    except Exception:
        pass

    if not HAS_TIFFFILE:
        raise RuntimeError(
            f"Could not open '{path}' with PIL and tifffile is not "
            f"installed. Run: pip install tifffile"
        )
    return tifffile.imread(path)


# ============================================================
# 4. AUTOMATIC RELIEF SCALING (data-driven, never hardcoded)
# ============================================================

def calculate_relief_scale(elevation, mesh_width, mesh_height, target_pct: float = 0.20):
    """
    target_pct controls how tall the visible relief should be relative
    to the mesh's horizontal size. Lowered default (0.20) for a gentler
    "uplifted" look rather than extreme spiky exaggeration.
    """
    p2 = np.percentile(elevation, 2)
    p98 = np.percentile(elevation, 98)
    relief = float(p98 - p2)

    if relief <= 0:
        return 1.0

    horizontal_size = max(mesh_width, mesh_height)
    target_relief = horizontal_size * target_pct
    scale = target_relief / relief

    scale = max(scale, 1.5)
    scale = min(scale, 18.0)

    return float(scale)


# ============================================================
# 5. CREATE SHARED X/Y
# ============================================================

def create_xy_grid(height, width, step=2):
    rows = np.arange(0, height, step)
    cols = np.arange(0, width, step)

    x = cols - (width - 1) / 2.0
    y = (height - 1) / 2.0 - rows

    X, Y = np.meshgrid(x, y)
    return X.astype(np.float32), Y.astype(np.float32)


# ============================================================
# 6. CREATE SAME FACE TOPOLOGY
# ============================================================

def create_faces(mesh_height, mesh_width):
    faces = []
    for r in range(mesh_height - 1):
        for c in range(mesh_width - 1):
            i = r * mesh_width + c
            faces.append([i, i + 1, i + mesh_width])
            faces.append([i + 1, i + mesh_width + 1, i + mesh_width])

    return np.asarray(faces, dtype=np.int64)


# ============================================================
# 7. CREATE UV
# ============================================================

def create_uv(height, width):
    u = np.linspace(0.0, 1.0, width)
    v = np.linspace(1.0, 0.0, height)

    U, V = np.meshgrid(u, v)
    return np.column_stack([U.ravel(), V.ravel()]).astype(np.float32)


# ============================================================
# 8. CREATE Z FROM ELEVATION (nothing else)
# ============================================================

def create_z(elevation, reference, scale):
    Z = (elevation - reference) * scale
    return Z.astype(np.float32)


# ============================================================
# 9. BUILD MESH
# ============================================================

def build_mesh(X, Y, Z, faces, texture):
    # glTF/GLB always treats Y as the "up" axis (X = right, Z = depth).
    vertices = np.column_stack([X.ravel(), Z.ravel(), Y.ravel()]).astype(np.float32)

    height, width = X.shape
    uv = create_uv(height, width)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(texture),
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )

    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    mesh.fix_normals()

    return mesh


# ============================================================
# 10. MAIN GENERATOR (call this directly from a backend)
# ============================================================

def generate_glbs(
    image_before,
    image_during,
    elevation_before,
    elevation_during,
    output_before="output/before.glb",
    output_during="output/during.glb",
    step=2,
    clip_low_pct=1.0,
    clip_high_pct=99.0,
    relief_target_pct=0.20,
    fix_tilt=False,
    border_trim_pct=DEFAULT_BORDER_TRIM_PCT,
    spike_smooth_sigma=1.2,
    min_feature_px=None,
    mad_multiplier=5.0,
    mountain_detail_boost=1.3,
):
    """
    Every argument accepts EITHER a file path (str) OR an already-loaded
    numpy array (for elevation) / PIL-loadable path or array (for
    images) -- so a backend can call this directly with in-memory data
    from an upload, no temp files required.
    """

    print("\n========================================")
    print("AUTOMATIC 3D RECONSTRUCTION")
    print("========================================")

    # ---- load elevation ----
    print("\nLoading elevation data...")
    before = load_elevation(elevation_before)
    during = load_elevation(elevation_during)
    before, during = align_shapes(before, during)

    # ---- load textures ----
    print("Loading RGB images...")
    texture_before = load_texture(image_before)
    texture_during = load_texture(image_during)

    # ---- trim tile-edge artifact border (elevation + texture, in sync) ----
    if border_trim_pct > 0:
        print(f"Trimming {border_trim_pct*100:.1f}% border margin "
              f"(removes DSM tile-edge 'collapsed wall' artifacts)...")
        before = auto_trim_border(before, border_trim_pct)
        during = auto_trim_border(during, border_trim_pct)
        texture_before = auto_trim_border(texture_before, border_trim_pct)
        texture_during = auto_trim_border(texture_during, border_trim_pct)

    H, W = before.shape

    # ---- clip DEM border noise ----
    print("Clipping outlier elevation values (DEM border noise)...")
    before = clip_elevation_outliers(before, clip_low_pct, clip_high_pct)
    during = clip_elevation_outliers(during, clip_low_pct, clip_high_pct)

    if fix_tilt:
        print("Removing systematic tilt (photogrammetry artifact)...")
        before = remove_planar_tilt(before)
        during = remove_planar_tilt(during)

    # ---- ground-flatten + keep buildings uplifted (not spiky) ----
    print("Separating terrain base from buildings/detail (with spike cleanup)...")

    # ---- DEBUG: mountain-region stats BEFORE separate_relief ----
    Hc, Wc = during.shape
    r0, r1 = int(Hc * 0.3), int(Hc * 0.9)
    c0, c1 = int(Wc * 0.55), Wc
    mountain_before = during[r0:r1, c0:c1]
    print(f"\n[DEBUG] DURING mountain region (rows {r0}:{r1}, cols {c0}:{c1}) "
          f"BEFORE separate_relief:")
    print(f"[DEBUG]   min={mountain_before.min():.3f} max={mountain_before.max():.3f} "
          f"relief(p2-p98)={np.percentile(mountain_before,98)-np.percentile(mountain_before,2):.3f}")

    before = separate_relief(before, spike_smooth_sigma, min_feature_px, mad_multiplier, mountain_detail_boost)
    during = separate_relief(during, spike_smooth_sigma, min_feature_px, mad_multiplier, mountain_detail_boost)

    # ---- DEBUG: mountain-region stats AFTER separate_relief ----
    mountain_after = during[r0:r1, c0:c1]
    print(f"[DEBUG] DURING mountain region AFTER separate_relief:")
    print(f"[DEBUG]   min={mountain_after.min():.3f} max={mountain_after.max():.3f} "
          f"relief(p2-p98)={np.percentile(mountain_after,98)-np.percentile(mountain_after,2):.3f}")

    # ---- downsample with area-average (no random patchy gaps) ----
    before_small = block_mean_downsample(before, step)
    during_small = block_mean_downsample(during, step)

    mesh_H, mesh_W = before_small.shape
    during_small = during_small[:mesh_H, :mesh_W]
    before_small = before_small[:mesh_H, :mesh_W]

    # ---- shared X/Y + shared faces ----
    print("\nCreating shared X/Y coordinates...")
    X, Y = create_xy_grid(H, W, step)
    X = X[:mesh_H, :mesh_W]
    Y = Y[:mesh_H, :mesh_W]
    faces = create_faces(mesh_H, mesh_W)

    # ---- common reference + automatic scale ----
    combined = np.concatenate([before_small.ravel(), during_small.ravel()])
    reference = np.percentile(combined, 2)

    scale_before = calculate_relief_scale(before_small, mesh_W, mesh_H, relief_target_pct)
    scale_during = calculate_relief_scale(during_small, mesh_W, mesh_H, relief_target_pct)
    common_scale = (scale_before + scale_during) / 2.0

    print("\nAutomatic vertical scale:", round(common_scale, 3))
    print("Elevation range BEFORE:", round(float(before_small.min()), 3),
          "to", round(float(before_small.max()), 3))
    print("Elevation range DURING:", round(float(during_small.min()), 3),
          "to", round(float(during_small.max()), 3))

    # ---- DEBUG: mountain-region stats AFTER downsampling, before Z scale ----
    r0s, r1s = int(mesh_H * 0.3), int(mesh_H * 0.9)
    c0s, c1s = int(mesh_W * 0.55), mesh_W
    mountain_downsampled = during_small[r0s:r1s, c0s:c1s]
    print(f"\n[DEBUG] DURING mountain region AFTER downsample (step={step}):")
    print(f"[DEBUG]   min={mountain_downsampled.min():.3f} max={mountain_downsampled.max():.3f} "
          f"relief(p2-p98)={np.percentile(mountain_downsampled,98)-np.percentile(mountain_downsampled,2):.3f}")
    print(f"[DEBUG]   common_scale being applied: {common_scale:.3f}")
    print(f"[DEBUG]   -> final Z relief in mountain region after scale: "
          f"{(np.percentile(mountain_downsampled,98)-np.percentile(mountain_downsampled,2)) * common_scale:.3f}")

    # ---- Z ----
    print("\nGenerating Z coordinates...")
    Z_before = create_z(before_small, reference, common_scale)
    Z_during = create_z(during_small, reference, common_scale)

    # ---- build meshes ----
    print("\nBuilding BEFORE mesh...")
    mesh_before = build_mesh(X, Y, Z_before, faces, texture_before)

    print("\nBuilding DURING mesh...")
    mesh_during = build_mesh(X, Y, Z_during, faces, texture_during)

    # ---- validation ----
    print("\nChecking geometry...")
    if not np.allclose(mesh_before.vertices[:, 0], mesh_during.vertices[:, 0]):
        raise ValueError("X coordinates are different!")
    if not np.allclose(mesh_before.vertices[:, 2], mesh_during.vertices[:, 2]):
        raise ValueError("Z (depth/north-south) coordinates are different!")
    if not np.array_equal(mesh_before.faces, mesh_during.faces):
        raise ValueError("Face topology is different!")

    print("\u2713 X coordinates SAME")
    print("\u2713 Z (depth/north-south) coordinates SAME")
    print("\u2713 Faces SAME")
    print("\u2713 Z coordinates from elevation only")

    # ---- output ----
    os.makedirs(os.path.dirname(output_before) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_during) or ".", exist_ok=True)

    print("\nExporting BEFORE GLB...")
    mesh_before.export(output_before, file_type="glb")
    print("Saved:", output_before)

    print("\nExporting DURING GLB...")
    mesh_during.export(output_during, file_type="glb")
    print("Saved:", output_during)

    print("\n========================================")
    print("3D RECONSTRUCTION COMPLETE")
    print("========================================")
    print("Vertices:", len(mesh_before.vertices))
    print("Faces:", len(mesh_before.faces))

    return {
        "before_glb": output_before,
        "during_glb": output_during,
        "vertex_count": len(mesh_before.vertices),
        "face_count": len(mesh_before.faces),
        "vertical_scale": common_scale,
    }


# ============================================================
# 11. TERMINAL INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Automatic Before/During 3D Terrain / DSM Generator"
    )

    parser.add_argument("--img-before", required=True)
    parser.add_argument("--img-during", required=True)
    parser.add_argument("--elev-before", required=True)
    parser.add_argument("--elev-during", required=True)
    parser.add_argument("--output", default="output")
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--clip-low-pct", type=float, default=1.0)
    parser.add_argument("--clip-high-pct", type=float, default=99.0)
    parser.add_argument("--relief-target-pct", type=float, default=0.20,
                         help="how tall terrain relief should look relative to model width (0.20 = 20%%). "
                              "Raise this (e.g. 0.3-0.4) for taller relief, lower it (e.g. 0.1-0.15) for "
                              "an even gentler look.")
    parser.add_argument("--fix-tilt", action="store_true",
                         help="remove systematic planar tilt (only use for flat-ground scenes with a "
                              "diagonal warp artifact -- do NOT use for mountain-dominant scenes)")
    parser.add_argument("--border-trim-pct", type=float, default=DEFAULT_BORDER_TRIM_PCT,
                         help=f"fraction of the grid trimmed off EVERY edge (default {DEFAULT_BORDER_TRIM_PCT})")
    parser.add_argument("--spike-smooth-sigma", type=float, default=1.2,
                         help="softens building/detail peaks so they look uplifted, not needle-sharp "
                              "(0 = off, higher = softer/rounder tops)")
    parser.add_argument("--min-feature-px", type=int, default=None,
                         help="minimum pixel width for a bump to count as a real building/feature "
                              "(smaller isolated spikes are removed as noise). Default: auto-scaled "
                              "from image resolution.")
    parser.add_argument("--mad-multiplier", type=float, default=5.0,
                         help="how conservative the noise-vs-feature threshold is (higher = stricter, "
                              "fewer things count as real features -- raise this if the WHOLE surface "
                              "still looks spiky/noisy, not just buildings)")
    parser.add_argument("--mountain-detail-boost", type=float, default=1.3,
                         help="amplifies mountain ridge/slope definition so terrain clearly reads as a "
                              "mountain (not a smooth blob). 1.0 = no boost, higher = more defined ridges.")

    args = parser.parse_args()

    output_before = os.path.join(args.output, "before.glb")
    output_during = os.path.join(args.output, "during.glb")

    generate_glbs(
        image_before=args.img_before,
        image_during=args.img_during,
        elevation_before=args.elev_before,
        elevation_during=args.elev_during,
        output_before=output_before,
        output_during=output_during,
        step=args.step,
        clip_low_pct=args.clip_low_pct,
        clip_high_pct=args.clip_high_pct,
        relief_target_pct=args.relief_target_pct,
        fix_tilt=args.fix_tilt,
        border_trim_pct=args.border_trim_pct,
        spike_smooth_sigma=args.spike_smooth_sigma,
        min_feature_px=args.min_feature_px,
        mad_multiplier=args.mad_multiplier,
        mountain_detail_boost=args.mountain_detail_boost,
    )