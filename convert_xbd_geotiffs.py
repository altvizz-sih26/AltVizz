"""
convert_xbd_geotiffs.py

Converts xBD PNG images (no embedded georeferencing) into proper GeoTIFFs,
using the geotransform data xBD provides in a separate "geotransforms.json"
file. Once converted, these work directly with process_new_image.py and
validate_spatial_alignment.py - no other changes needed anywhere else.

The geotransforms JSON maps filename -> [gdal_geotransform_6_values, wkt_crs_string]
e.g. "woolsey-fire_00000877_pre_disaster.png": [[-118.82..., 4.57e-06, 0.0, 34.11..., 0.0, -4.57e-06], "GEOGCS[...]"]

Usage:
    python convert_xbd_geotiffs.py geotransforms.json png_folder output_folder
"""
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.transform import Affine
from PIL import Image


def convert_one(png_path, geotransform, wkt_crs, out_path):
    img = Image.open(png_path)
    arr = np.array(img)

    if arr.ndim == 2:
        arr = arr[:, :, None]

    height, width, bands = arr.shape
    transform = Affine.from_gdal(*geotransform)

    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=height, width=width, count=bands,
        dtype=arr.dtype, crs=wkt_crs, transform=transform,
    ) as dst:
        for b in range(bands):
            dst.write(arr[:, :, b], b + 1)


def main():
    if len(sys.argv) != 4:
        print("Usage: python convert_xbd_geotiffs.py geotransforms.json png_folder output_folder")
        sys.exit(1)

    geotransforms_path = sys.argv[1]
    png_folder = sys.argv[2]
    output_folder = sys.argv[3]

    with open(geotransforms_path, "r") as f:
        geo_data = json.load(f)

    os.makedirs(output_folder, exist_ok=True)

    converted = 0
    skipped = 0
    for filename in geo_data:
        entry = geo_data[filename]
        geotransform = entry[0]
        wkt_crs = entry[1]

        png_path = os.path.join(png_folder, filename)
        if not os.path.isfile(png_path):
            skipped = skipped + 1
            continue

        out_name = os.path.splitext(filename)[0] + ".tif"
        out_path = os.path.join(output_folder, out_name)
        try:
            convert_one(png_path, geotransform, wkt_crs, out_path)
            converted = converted + 1
            print("Converted: " + filename)
        except Exception as e:
            print("FAILED: " + filename + " (" + str(e) + ")")
            skipped = skipped + 1

    print("")
    print("Done. Converted: " + str(converted) + ", Skipped/failed: " + str(skipped))


if __name__ == "__main__":
    main()
