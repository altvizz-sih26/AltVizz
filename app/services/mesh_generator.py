"""
Turns a calibrated elevation grid + the original source image into a real
3D terrain mesh (.glb), instead of the 3 fixed pre-made demo files.

Approach mirrors the pre-made demo GLBs already in public/ (per main.js's
own comments: Z-up heightfield, single material, baked satellite texture,
POSITION + TEXCOORD_0 only) - so the existing viewer's material/axis-fix
code in main.js should handle these newly-generated files the same way
it already handles the pre-made ones, with no viewer changes needed for
material correctness.
"""
import cv2
import numpy as np
import trimesh
from PIL import Image

# Keep mesh size reasonable (matches the ~512x512 / ~262k-vertex scale of
# the pre-made demo GLBs) - higher resolution makes bigger, slower-to-load
# files without much visible benefit in a hackathon demo.
DEFAULT_RESOLUTION = 256

# Footprint half-size in mesh units - arbitrary but matches the rough
# scale (~511 units) the existing viewer's camera framing already expects.
FOOTPRINT_HALF_SIZE = 255.0


def generate_glb(elevation: np.ndarray, texture_image_bgr: np.ndarray, out_path: str,
                  resolution: int = DEFAULT_RESOLUTION) -> None:
    """
    Builds a heightfield mesh from `elevation` (2D array, meters or
    relative units), textures it with `texture_image_bgr` (the original
    uploaded image), and writes a .glb file to `out_path`.
    """
    # Downsample both to the same working resolution
    elev = cv2.resize(elevation.astype(np.float32), (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    elev = cv2.GaussianBlur(elev, (9, 9), 0)  # softens harsh jumps between misclassified terrain patches
    texture_rgb = cv2.cvtColor(
        cv2.resize(texture_image_bgr, (resolution, resolution), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB,
    )

    rows, cols = elev.shape

    # --- Vertical exaggeration ---
    # Real elevation ranges (even calibrated ones) are often small relative
    # to the horizontal footprint, which would look nearly flat. Scale the
    # relief to a fixed fraction of the horizontal footprint so it's
    # visually readable, regardless of whether the input is a 0-50m
    # "relative" range or a real 223-267m SRTM-calibrated range.
    elev_range = float(np.nanmax(elev) - np.nanmin(elev))
    target_relief = FOOTPRINT_HALF_SIZE * 0.15  # relief reaches ~35% of the footprint's half-size
    exaggeration = (target_relief / elev_range) if elev_range > 1e-6 else 1.0
    z = (elev - np.nanmin(elev)) * exaggeration

    # --- Build vertex grid (Z-up, matches the pre-made GLBs' authoring convention) ---
    xs = np.linspace(-FOOTPRINT_HALF_SIZE, FOOTPRINT_HALF_SIZE, cols)
    ys = np.linspace(-FOOTPRINT_HALF_SIZE, FOOTPRINT_HALF_SIZE, rows)
    xx, yy = np.meshgrid(xs, ys)

    vertices = np.column_stack([xx.ravel(), yy.ravel(), z.ravel()])

    # UVs: standard top-left-origin image mapping
    us = np.linspace(0, 1, cols)
    vs = np.linspace(1, 0, rows)
    uu, vv = np.meshgrid(us, vs)
    uvs = np.column_stack([uu.ravel(), vv.ravel()])

    # --- Build faces: two triangles per grid cell ---
    faces = []
    for r in range(rows - 1):
        row_start = r * cols
        next_row_start = (r + 1) * cols
        for c in range(cols - 1):
            v0 = row_start + c
            v1 = row_start + c + 1
            v2 = next_row_start + c
            v3 = next_row_start + c + 1
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])
    faces = np.array(faces)

    # --- Bake the texture and export ---
    pil_image = Image.fromarray(texture_rgb)
    visual = trimesh.visual.TextureVisuals(uv=uvs, image=pil_image)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)

    mesh.export(out_path, file_type="glb")