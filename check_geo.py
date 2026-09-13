import rasterio

files = {
    "urban": "test_data/urban/urban_1.tif",
    "vegetation": "test_data/vegetation/vegetation_1.tif",
    "bare_terrain": "test_data/bare_terrain/bare_terrain_1.tif",
}

for name, path in files.items():
    print(f"\n--- {name} ---")
    try:
        with rasterio.open(path) as src:
            print("CRS:", src.crs)
            print("Bounds:", src.bounds)
            print("Width x Height:", src.width, "x", src.height)
    except Exception as e:
        print("ERROR:", e)