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
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds, reproject, Resampling
import numpy as np

API_KEY = os.getenv("OPENTOPO_API_KEY", "")          # your OpenTopography API key
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
    api_key = os.getenv("OPENTOPO_API_KEY", API_KEY).strip()
    if not api_key:
        raise RuntimeError("SRTM unavailable: OPENTOPO_API_KEY is not configured")

    print(f"Downloading SRTM for bounds: {lon_min:.4f}, {lat_min:.4f}, {lon_max:.4f}, {lat_max:.4f}")
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": "SRTMGL1",       # SRTM 30m
        "south": lat_min,
        "north": lat_max,
        "west": lon_min,
        "east": lon_max,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    response = requests.get(url, params=params, timeout=(10, 90))
    if response.status_code != 200:
        raise RuntimeError(
            f"SRTM unavailable: OpenTopography returned HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )
    if len(response.content) <= 1000:
        raise RuntimeError("SRTM unavailable: OpenTopography returned an empty response")

    try:
        with MemoryFile(response.content) as memory_file:
            with memory_file.open() as dataset:
                if dataset.count < 1 or dataset.crs is None:
                    raise ValueError("response has no raster band or coordinate system")
                dataset.read(1, window=((0, 1), (0, 1)))
    except Exception as error:
        raise RuntimeError(f"SRTM unavailable: response was not a valid GeoTIFF: {error}") from error

    with open(out_path, "wb") as output:
        output.write(response.content)
    print(f"Saved raw SRTM: {out_path} ({len(response.content)} bytes)")
    return out_path


def align_srtm_to_image(image_path, srtm_path, out_path, bounds_wgs84=None):
    """Step 3: reproject SRTM onto the image's exact pixel grid, with proper
    nodata handling (this part was already generic — works for any image)."""
    with rasterio.open(image_path) as ref:
        ref_width = ref.width
        ref_height = ref.height
        if bounds_wgs84 is None:
            ref_crs = ref.crs
            ref_transform = ref.transform
            if ref_crs is None:
                raise ValueError(f"{image_path} has no CRS; provide a location to align SRTM")
        else:
            ref_crs = CRS.from_epsg(4326)
            ref_transform = from_bounds(*bounds_wgs84, ref_width, ref_height)

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


def process_new_image(
    image_path,
    output_dir=None,
    center_lat=None,
    center_lon=None,
    ground_width_m=None,
):
    """Full pipeline: any image in -> aligned SRTM reference out."""
    output_dir = output_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    raw_srtm_path = os.path.join(output_dir, f"srtm_{basename}_raw.tif")
    aligned_srtm_path = os.path.join(output_dir, f"srtm_{basename}_aligned.tif")

    print(f"\n=== Processing: {image_path} ===\n")

    location = (center_lat, center_lon, ground_width_m)
    if any(value is not None for value in location):
        if any(value is None for value in location):
            raise ValueError("Latitude, longitude, and ground width must be supplied together")
        if not -90 < center_lat < 90 or not -180 <= center_lon <= 180 or ground_width_m <= 0:
            raise ValueError("Location must use valid latitude/longitude and a positive ground width")
        with rasterio.open(image_path) as image:
            ground_height_m = ground_width_m * image.height / image.width
        lat_radius = ground_height_m / (2 * 111320.0)
        lon_radius = ground_width_m / (
            2 * 111320.0 * max(abs(np.cos(np.deg2rad(center_lat))), 1e-6)
        )
        lon_min, lon_max = center_lon - lon_radius, center_lon + lon_radius
        lat_min, lat_max = center_lat - lat_radius, center_lat + lat_radius
        if lat_min < -90 or lat_max > 90:
            raise ValueError("Ground-width extent extends beyond valid latitude bounds")
        bounds_wgs84 = (lon_min, lat_min, lon_max, lat_max)
    else:
        bounds_wgs84 = None
        lon_min, lat_min, lon_max, lat_max = get_image_bounds_wgs84(image_path)

    print(f"Detected bounds (lon/lat): {lon_min:.4f}, {lat_min:.4f}, {lon_max:.4f}, {lat_max:.4f}\n")

    download_srtm(lon_min, lat_min, lon_max, lat_max, raw_srtm_path)
    align_srtm_to_image(image_path, raw_srtm_path, aligned_srtm_path, bounds_wgs84)

    print(f"\nDone. Reference elevation ready at: {aligned_srtm_path}")
    return aligned_srtm_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_new_image.py path/to/any_image.tif")
        sys.exit(1)
    process_new_image(sys.argv[1])