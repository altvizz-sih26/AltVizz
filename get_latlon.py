import rasterio
from rasterio.warp import transform_bounds

files = {
    "urban": "test_data/urban/urban_1.tif",
    "vegetation": "test_data/vegetation/vegetation_1.tif",
    "bare_terrain": "test_data/bare_terrain/bare_terrain_1.tif",
}

for name, path in files.items():
    with rasterio.open(path) as src:
        lon_min, lat_min, lon_max, lat_max = transform_bounds(
            src.crs, "EPSG:4326", *src.bounds
        )
        print(f"\n--- {name} ---")
        print(f"Lon: {lon_min:.4f} to {lon_max:.4f}")
        print(f"Lat: {lat_min:.4f} to {lat_max:.4f}")