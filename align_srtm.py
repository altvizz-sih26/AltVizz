import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

pairs = {
    "urban": ("test_data/urban/urban_1.tif", "test_data/urban/srtm_urban.tif"),
    "vegetation": ("test_data/vegetation/vegetation_1.tif", "test_data/vegetation/srtm_vegetation.tif"),
    "bare_terrain": ("test_data/bare_terrain/bare_terrain_1.tif", "test_data/bare_terrain/srtm_bare_terrain.tif"),
}

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

        dst_data = np.empty((ref_height, ref_width), dtype=np.float32)

        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
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
        }

        out_path = f"test_data/{name}/srtm_{name}_aligned.tif"
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(dst_data, 1)

        print(f"Saved: {out_path}")
        print(f"Shape: {dst_data.shape}, Min: {dst_data.min():.1f}m, Max: {dst_data.max():.1f}m")