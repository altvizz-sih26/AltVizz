import rasterio

with rasterio.open("person2_elevation/N24W078.hgt") as dataset:
    print("Width, Height:", dataset.width, dataset.height)
    print("CRS:", dataset.crs)
    print("Bounds:", dataset.bounds)
    
    elevation_data = dataset.read(1)
    
    print("Min elevation (meters):", elevation_data.min())
    print("Max elevation (meters):", elevation_data.max())
    print("Mean elevation (meters):", elevation_data.mean())