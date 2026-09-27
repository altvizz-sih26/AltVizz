import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

pairs = {
    "urban": ("test_data/urban/urban_1.tif", "test_data/urban/srtm_urban.tif"),
    "vegetation": ("test_data/vegetation/vegetation_1.tif", "test_data/vegetation/srtm_vegetation.tif"),
    "bare_terrain": ("test_data/bare_terrain/bare_terrain_1.tif", "test_data/bare_terrain/srtm_bare_terrain.tif"),
}

NODATA_VALUE = -9999.0  # sentinel for "no real data here" instead of 0.0

for name, (img_path, srtm_path) in pairs.items():
    print(f"\nAligning {name}...")
    with rasterio.open(img_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height

    with rasterio.open(srtm_path) as src:
        src_data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs

        # fill destination with NODATA first, so untouched edge pixels stay NODATA
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

        out_path = f"test_data/{name}/srtm_{name}_aligned.tif"
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(dst_data, 1)

        valid = dst_data[dst_data != NODATA_VALUE]
        print(f"Saved: {out_path}")
        print(f"Shape: {dst_data.shape}, Valid pixels: {valid.size}/{dst_data.size}")