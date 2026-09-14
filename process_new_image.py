"""
process_new_image.py

GENERIC version — works for ANY georeferenced input image, not just the 3 known
test regions. Given any GeoTIFF, this automatically:
  1. Reads the image's own bounds (no manual lat/lon typing needed)
  2. Downloads the matching SRTM elevation tile from OpenTopography
  3. Reprojects/aligns that SRTM data onto the image's exact pixel grid,
     with proper nodata handling for edge artifacts

Output: an aligned SRTM GeoTIFF, pixel-for-pixel matched to your input image,
ready to be used as validation reference for that image's model output.

Usage:
    python process_new_image.py path/to/any_image.tif
"""

import sys
import os
import requests
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
import numpy as np


NODATA_VALUE = -9999.0             # sentinel for "no real SRTM data here"


def get_image_bounds_wgs84(image_path):
    """Step 1: read the image's own bounds and convert to lat/lon (WGS84) —
    no manual coordinate entry needed, works for any georeferenced image."""
    with rasterio.open(image_path) as src:
        if src.crs is None:
            raise ValueError(
                f"{image_path} has no CRS/georeferencing — this only works "
                "on georeferenced images (GeoTIFF with spatial metadata)."
            )
        lon_min, lat_min, lon_max, lat_max = transform_bounds(
            src.crs, "EPSG:4326", *src.bounds
        )
        return lon_min, lat_min, lon_max, lat_max


def download_srtm(lon_min, lat_min, lon_max, lat_max, out_path):
    """Step 2: download SRTM covering exactly this bounding box."""
    print(f"Downloading SRTM for bounds: {lon_min:.4f}, {lat_min:.4f}, {lon_max:.4f}, {lat_max:.4f}")
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": "SRTMGL1",       # SRTM 30m
        "south": lat_min,
        "north": lat_max,
        "west": lon_min,
        "east": lon_max,
        "outputFormat": "GTiff",
        "API_Key": API_KEY,
    }
    r = requests.get(url, params=params)
    if r.status_code == 200 and len(r.content) > 1000:
        with open(out_path, "wb") as f:
            f.write(r.content)
        print(f"Saved raw SRTM: {out_path} ({len(r.content)} bytes)")
        return out_path
    else:
        raise RuntimeError(f"SRTM download failed: status={r.status_code}\n{r.text[:500]}")


def align_srtm_to_image(image_path, srtm_path, out_path):
    """Step 3: reproject SRTM onto the image's exact pixel grid, with proper
    nodata handling (this part was already generic — works for any image)."""
    with rasterio.open(image_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height

    with rasterio.open(srtm_path) as src:
        src_data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs

        dst_data = np.full((ref_height, ref_width), NODATA_VALUE, dtype=np.float32)

        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=NODATA_VALUE,
            resampling=Resampling.bilinear,
        )

        out_meta = {
            "driver": "GTiff",
            "height": ref_height,
            "width": ref_width,
            "count": 1,
            "dtype": "float32",
            "crs": ref_crs,
            "transform": ref_transform,
            "nodata": NODATA_VALUE,
        }

        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(dst_data, 1)

        valid = dst_data[dst_data != NODATA_VALUE]
        print(f"Saved aligned SRTM: {out_path}")
        print(f"Shape: {dst_data.shape}, Valid pixels: {valid.size}/{dst_data.size}")
        if valid.size > 0:
            print(f"Elevation range: {valid.min():.1f}m to {valid.max():.1f}m")


def process_new_image(image_path):
    """Full pipeline: any image in -> aligned SRTM reference out."""
    basename = os.path.splitext(os.path.basename(image_path))[0]
    raw_srtm_path = f"srtm_{basename}_raw.tif"
    aligned_srtm_path = f"srtm_{basename}_aligned.tif"

    print(f"\n=== Processing: {image_path} ===\n")

    lon_min, lat_min, lon_max, lat_max = get_image_bounds_wgs84(image_path)
    print(f"Detected bounds (lon/lat): {lon_min:.4f}, {lat_min:.4f}, {lon_max:.4f}, {lat_max:.4f}\n")

    download_srtm(lon_min, lat_min, lon_max, lat_max, raw_srtm_path)
    align_srtm_to_image(image_path, raw_srtm_path, aligned_srtm_path)

    print(f"\nDone. Reference elevation ready at: {aligned_srtm_path}")
    return aligned_srtm_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_new_image.py path/to/any_image.tif")
        sys.exit(1)
    process_new_image(sys.argv[1])