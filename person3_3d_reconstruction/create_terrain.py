"""
create_terrain.py (generic version)

Converts an elevation array + its source satellite image into a textured
3D mesh, exported as .glb — for ANY region.

Supports:
    - GeoTIFF / TIFF images
    - PNG images
    - JPG / JPEG images

Standalone usage:
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
        elevation: 2D numpy array of elevation values.
        image_path: Path to the source satellite image.
        output_path: Where to write the resulting .glb file.
        step: Downsampling stride.
        vertical_exaggeration: Multiplier for visual height exaggeration.
        smooth_sigma: Gaussian smoothing applied to elevation.

    Returns:
        output_path, on success.
    """

    elevation = elevation.astype(np.float32)

    # --- Remove invalid elevation values -------------------------------
    elevation = np.nan_to_num(
        elevation,
        nan=np.nanmedian(elevation),
        posinf=np.nanmax(elevation),
        neginf=np.nanmin(elevation),
    )

    # --- Load satellite image -------------------------------------------
    #
    # TIFF/GeoTIFF files need tifffile because they may contain multiple
    # bands or use channel-first layouts.
    #
    # PNG/JPG/JPEG files are standard RGB images, so PIL is sufficient.
    #
    if image_path.lower().endswith((".tif", ".tiff")):
        image = tifffile.imread(image_path)

        if image.ndim == 3:
            if image.shape[2] >= 3:
                # H x W x bands
                rgb = image[:, :, :3]

            elif image.shape[0] >= 3:
                # bands x H x W
                rgb = np.transpose(image[:3], (1, 2, 0))

            else:
                raise ValueError("Could not find RGB channels in TIFF image.")

        else:
            raise ValueError("Unexpected TIFF satellite image format.")

    else:
        # PNG / JPG / JPEG
        rgb = np.array(Image.open(image_path).convert("RGB"))

    # --- Normalize RGB values -------------------------------------------
    rgb = rgb.astype(np.float32)

    if rgb.max() > 255:
        rgb = (rgb / rgb.max()) * 255

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # --- Match elevation + image dimensions ----------------------------
    H0 = min(elevation.shape[0], rgb.shape[0])
    W0 = min(elevation.shape[1], rgb.shape[1])

    elevation = elevation[:H0, :W0]
    rgb = rgb[:H0, :W0]

    # --- Downsample ------------------------------------------------------
    elevation = elevation[::step, ::step]
    rgb = rgb[::step, ::step]

    H, W = elevation.shape

    # --- Smooth elevation -----------------------------------------------
    elevation_smooth = gaussian_filter(
        elevation,
        sigma=smooth_sigma,
    )

    # --- Convert to relative height ------------------------------------
    base = np.percentile(elevation_smooth, 2)

    relative_height = (
        elevation_smooth - base
    ) * vertical_exaggeration

    # --- Create grid -----------------------------------------------------
    x = np.arange(W) - (W - 1) / 2
    y = np.arange(H) - (H - 1) / 2

    X, Y = np.meshgrid(x, -y)
    Z = relative_height

    vertices = np.column_stack(
        (
            X.ravel(),
            Y.ravel(),
            Z.ravel(),
        )
    ).astype(np.float32)

    # --- Create triangles -----------------------------------------------
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

    faces = np.asarray(
        faces,
        dtype=np.int32,
    )

    # --- Create mesh -----------------------------------------------------
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )

    # --- UV coordinates + texture --------------------------------------
    u = np.linspace(0, 1, W)
    v = np.linspace(1, 0, H)

    U, V = np.meshgrid(u, v)

    uv = np.column_stack(
        (
            U.ravel(),
            V.ravel(),
        )
    ).astype(np.float32)

    texture = Image.fromarray(rgb)

    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=uv,
        image=texture,
    )

    mesh.fix_normals()

    # --- Export GLB ------------------------------------------------------
    mesh.export(output_path)

    return output_path


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print(
            "Usage: python create_terrain.py "
            "<elevation.npy> <image> <output.glb>"
        )
        sys.exit(1)

    elevation_file = sys.argv[1]
    image_file = sys.argv[2]
    output_file = sys.argv[3]

    print(f"Loading elevation from {elevation_file}...")

    elev_array = np.load(
        elevation_file
    ).astype(np.float32)

    print(f"Generating mesh from {image_file}...")

    generate_glb(
        elev_array,
        image_file,
        output_file,
    )

    print(f"Saved: {output_file}")