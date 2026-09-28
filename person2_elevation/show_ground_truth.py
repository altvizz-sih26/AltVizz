import rasterio
import numpy as np
import matplotlib.pyplot as plt

with rasterio.open("person2_elevation/srtm_vegetation_aligned.tif") as ds:
    elevation = ds.read(1).astype(np.float32)

print(f"Shape: {elevation.shape}")
print(f"Elevation range: {elevation.min():.1f}m to {elevation.max():.1f}m")
print(f"Mean elevation: {elevation.mean():.1f}m")

plt.figure(figsize=(8, 8))
im = plt.imshow(elevation, cmap='terrain')
cbar = plt.colorbar(im, label='Elevation (meters)')
plt.title(f'Real SRTM Ground Truth Elevation\nvegetation_1 — Range: {elevation.min():.0f}m to {elevation.max():.0f}m')
plt.savefig("person2_elevation/proof_of_real_elevation_data.png", dpi=200)
plt.show()

print("\nSaved: proof_of_real_elevation_data.png")