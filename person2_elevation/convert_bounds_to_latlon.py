import rasterio
from rasterio.warp import transform_bounds

with rasterio.open("person2_elevation/RGB.byte.tif") as dataset:
    utm_bounds = dataset.bounds
    utm_crs = dataset.crs
    
    lonlat_bounds = transform_bounds(utm_crs, "EPSG:4326", *utm_bounds)
    
    print("Original UTM bounds:", utm_bounds)
    print("Converted lat/lon bounds (left, bottom, right, top):", lonlat_bounds)