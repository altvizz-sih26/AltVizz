import elevation

# Bounds from our previous script (left, bottom, right, top)
bounds = (-78.96, 23.56, -76.57, 25.55)

elevation.clip(bounds=bounds, output="person2_elevation/srtm_sample.tif")

print("SRTM data downloaded and saved.")