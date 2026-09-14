"""
create_terrain.py (generic version)

Converts an elevation array + its source satellite image into a textured
3D mesh, exported as .glb — for ANY region, not just a hardcoded one.

Same core logic as the original create_terrain.py (trimesh-based heightfield
mesh + satellite texture), refactored into a function so depth_pipeline.py
can call it directly on whatever elevation array + image it just produced,
instead of requiring hardcoded ELEVATION_FILE/IMAGE_FILE constants.

Standalone usage (unchanged from before, for manual testing):
    python create_terrain.py final_elevation_vegetation.npy vegetation_1.tif output.glb
"""

import sys

import numpy as np
import tifffile
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter


def generate_glb(
    elevation: np.ndarray,
    image_path: str,
    output_path: str,
    step: int = 2,
    vertical_exaggeration: float = 1.5,
    smooth_sigma: float = 1.2,
) -> str:
    """
    Builds a textured 3D terrain mesh from an elevation array + its source
    satellite image, and exports it as a .glb file.

    Args:
        elevation: 2D numpy array of elevation values (already computed —
            e.g. straight from Person 2's calibration step, no need to
            reload from a .npy file).
        image_path: path to the source satellite image (GeoTIFF/tif) whose
            RGB will be used as the mesh texture. Read fresh here (not
            reused from an already-loaded array) since satellite tifs can
            have >3 bands and need the same channel-extraction handling
            as the original script.
        output_path: where to write the resulting .glb file.
        step: downsampling stride (2 = half resolution per axis, keeps
            mesh size/browser performance reasonable).
        vertical_exaggeration: multiplier on relative height, for a more
            visually readable 3D effect.
        smooth_sigma: Gaussian smoothing applied to elevation before
            meshing, to remove pixel-level noise.

    Returns:
        output_path, on success.
    """
    elevation = elevation.astype(np.float32)

    # Remove invalid values
    elevation = np.nan_to_num(
        elevation,
        nan=np.nanmedian(elevation),
        posinf=np.nanmax(elevation),
        neginf=np.nanmin(elevation),
    )

    # --- Load satellite image -------------------------------------------
    image = tifffile.imread(image_path)

    if image.ndim == 3:
        if image.shape[2] >= 3:
            rgb = image[:, :, :3]
        elif image.shape[0] >= 3:
            rgb = np.transpose(image[:3], (1, 2, 0))
        else:
            raise ValueError("Could not find RGB channels.")
    else:
        raise ValueError("Unexpected satellite image format.")

    rgb = rgb.astype(np.float32)
    if rgb.max() > 255:
        rgb = (rgb / rgb.max()) * 255
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # --- Match elevation + image dimensions -------------------------------
    H0 = min(elevation.shape[0], rgb.shape[0])
    W0 = min(elevation.shape[1], rgb.shape[1])
    elevation = elevation[:H0, :W0]
    rgb = rgb[:H0, :W0]

    # --- Downsample ---------------------------------------------------------
    elevation = elevation[::step, ::step]
    rgb = rgb[::step, ::step]
    H, W = elevation.shape

    # --- Smooth elevation -----------------------------------------------
    elevation_smooth = gaussian_filter(elevation, sigma=smooth_sigma)

    # --- Convert to relative height (lowest point -> 0) --------------------
    base = np.percentile(elevation_smooth, 2)
    relative_height = (elevation_smooth - base) * vertical_exaggeration

    # --- Create grid -------------------------------------------------------
    x = np.arange(W) - (W - 1) / 2
    y = np.arange(H) - (H - 1) / 2
    X, Y = np.meshgrid(x, -y)
    Z = relative_height

    vertices = np.column_stack((X.ravel(), Y.ravel(), Z.ravel())).astype(np.float32)

    # --- Create triangles ----------------------------------------------
    faces = []
    for row in range(H - 1):
        current = row * W
        next_row = (row + 1) * W
        for col in range(W - 1):
            a = current + col
            b = current + col + 1
            c = next_row + col
            d = next_row + col + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    faces = np.asarray(faces, dtype=np.int32)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # --- UV coordinates + texture -----------------------------------------
    u = np.linspace(0, 1, W)
    v = np.linspace(1, 0, H)
    U, V = np.meshgrid(u, v)
    uv = np.column_stack((U.ravel(), V.ravel())).astype(np.float32)

    texture = Image.fromarray(rgb)
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=texture)
    mesh.fix_normals()

    mesh.export(output_path)
    return output_path


if __name__ == "__main__":
    # Standalone/manual usage, e.g.:
    #   python create_terrain.py final_elevation_vegetation.npy vegetation_1.tif output.glb
    if len(sys.argv) != 4:
        print("Usage: python create_terrain.py <elevation.npy> <image.tif> <output.glb>")
        sys.exit(1)

    elevation_file, image_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"Loading elevation from {elevation_file}...")
    elev_array = np.load(elevation_file).astype(np.float32)
    print("Generating mesh...")
    generate_glb(elev_array, image_file, output_file)
    print(f"Saved: {output_file}")