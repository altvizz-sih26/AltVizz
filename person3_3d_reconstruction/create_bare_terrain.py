import numpy as np
import tifffile
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter


# ==========================================================
# FILES
# ==========================================================

ELEVATION_FILE = "final_elevation_bare_terrain.npy"
IMAGE_FILE = "bare_terrain_1.tif"

OUTPUT_FILE = "depthwizard_bare_terrain_3d.glb"


# ==========================================================
# SETTINGS
# ==========================================================

# 1024 x 1024 -> 512 x 512
STEP = 2

# Bare terrain has a larger elevation range,
# so keep exaggeration moderate.
VERTICAL_EXAGGERATION = 5.0

# Smooth small elevation noise
SMOOTH_SIGMA = 0.5


# ==========================================================
# LOAD ELEVATION
# ==========================================================

print()
print("Loading bare terrain elevation...")

elevation = np.load(
    ELEVATION_FILE
).astype(np.float32)

print("Elevation shape:", elevation.shape)

print("Minimum elevation:",
      np.nanmin(elevation), "m")

print("Maximum elevation:",
      np.nanmax(elevation), "m")


# ==========================================================
# REMOVE INVALID VALUES
# ==========================================================

median_value = np.nanmedian(elevation)

elevation = np.nan_to_num(
    elevation,
    nan=median_value,
    posinf=np.nanmax(elevation),
    neginf=np.nanmin(elevation)
)


# ==========================================================
# LOAD ORIGINAL TERRAIN IMAGE
# ==========================================================

print()
print("Loading terrain image...")

image = tifffile.imread(
    IMAGE_FILE
)

print("Image shape:", image.shape)


# ==========================================================
# EXTRACT RGB
# ==========================================================

if image.ndim == 3:

    # H x W x Channels
    if image.shape[2] >= 3:

        rgb = image[:, :, :3]

    # Channels x H x W
    elif image.shape[0] >= 3:

        rgb = np.transpose(
            image[:3],
            (1, 2, 0)
        )

    else:

        raise ValueError(
            "RGB channels not found."
        )

else:

    raise ValueError(
        "Unexpected TIFF format."
    )


# ==========================================================
# CONVERT RGB TO UINT8
# ==========================================================

rgb = rgb.astype(
    np.float32
)

if rgb.max() > 255:

    rgb = (
        rgb / rgb.max()
    ) * 255

rgb = np.clip(
    rgb,
    0,
    255
).astype(
    np.uint8
)


# ==========================================================
# MATCH IMAGE AND ELEVATION SIZE
# ==========================================================

H0 = min(
    elevation.shape[0],
    rgb.shape[0]
)

W0 = min(
    elevation.shape[1],
    rgb.shape[1]
)

elevation = elevation[
    :H0,
    :W0
]

rgb = rgb[
    :H0,
    :W0
]


# ==========================================================
# DOWNSAMPLE
# ==========================================================

elevation = elevation[
    ::STEP,
    ::STEP
]

rgb = rgb[
    ::STEP,
    ::STEP
]

H, W = elevation.shape

print()
print(
    "3D mesh resolution:",
    W,
    "x",
    H
)


# ==========================================================
# SMOOTH ELEVATION
# ==========================================================

print()
print("Smoothing terrain...")

elevation_smooth = gaussian_filter(
    elevation,
    sigma=SMOOTH_SIGMA
)


# ==========================================================
# CONVERT ABSOLUTE ELEVATION TO RELATIVE HEIGHT
# ==========================================================

# Lowest terrain becomes approximately zero.
# Elevation differences are preserved.

base = np.percentile(
    elevation_smooth,
    2
)

relative_height = (
    elevation_smooth - base
)


# ==========================================================
# VERTICAL EXAGGERATION
# ==========================================================

relative_height *= (
    VERTICAL_EXAGGERATION
)


# ==========================================================
# CREATE X-Y GRID
# ==========================================================

print()
print("Creating 3D terrain grid...")

x = (
    np.arange(W)
    - (W - 1) / 2
)

y = (
    np.arange(H)
    - (H - 1) / 2
)

X, Y = np.meshgrid(
    x,
    -y
)

Z = relative_height


# ==========================================================
# CREATE VERTICES
# ==========================================================

vertices = np.column_stack(
    (
        X.ravel(),
        Y.ravel(),
        Z.ravel()
    )
).astype(
    np.float32
)


# ==========================================================
# CREATE TRIANGLES
# ==========================================================

print(
    "Creating triangular mesh..."
)

faces = []

for row in range(H - 1):

    current = row * W
    next_row = (row + 1) * W

    for col in range(W - 1):

        a = current + col
        b = current + col + 1

        c = next_row + col
        d = next_row + col + 1

        faces.append(
            [a, b, c]
        )

        faces.append(
            [b, d, c]
        )


faces = np.asarray(
    faces,
    dtype=np.int32
)


# ==========================================================
# CREATE MESH
# ==========================================================

print("Building mesh...")

mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    process=False
)


# ==========================================================
# CREATE UV COORDINATES
# ==========================================================

print(
    "Creating texture coordinates..."
)

u = np.linspace(
    0,
    1,
    W
)

v = np.linspace(
    1,
    0,
    H
)

U, V = np.meshgrid(
    u,
    v
)

uv = np.column_stack(
    (
        U.ravel(),
        V.ravel()
    )
).astype(
    np.float32
)


# ==========================================================
# APPLY ORIGINAL TERRAIN IMAGE
# ==========================================================

print(
    "Applying original terrain texture..."
)

texture = Image.fromarray(
    rgb
)

mesh.visual = (
    trimesh.visual.texture.TextureVisuals(
        uv=uv,
        image=texture
    )
)


# ==========================================================
# FIX NORMALS
# ==========================================================

mesh.fix_normals()


# ==========================================================
# EXPORT GLB
# ==========================================================

print()
print("Exporting GLB...")

mesh.export(
    OUTPUT_FILE
)


# ==========================================================
# COMPLETE
# ==========================================================

print()
print("==============================================")
print("     BARE TERRAIN 3D MODEL CREATED!")
print("==============================================")

print(
    "Output:",
    OUTPUT_FILE
)

print(
    "Resolution:",
    W,
    "x",
    H
)

print(
    "Vertical exaggeration:",
    VERTICAL_EXAGGERATION
)

print()