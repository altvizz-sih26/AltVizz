import rasterio

with rasterio.open("person2_elevation/RGB.byte.tif") as dataset:
    print("Width, Height:", dataset.width, dataset.height)
    print("CRS:", dataset.crs)
    print("Bounds:", dataset.bounds)
    print("Transform:", dataset.transform)
    print("Number of bands:", dataset.count)