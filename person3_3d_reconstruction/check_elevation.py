import numpy as np

file = "final_elevation_bare_terrain.npy"

elevation = np.load(file)

print("Shape:", elevation.shape)
print("Minimum:", elevation.min())
print("Maximum:", elevation.max())
print("Range:", elevation.max() - elevation.min())

print("\nPercentiles:")
print("1%  :", np.percentile(elevation, 1))
print("25% :", np.percentile(elevation, 25))
print("50% :", np.percentile(elevation, 50))
print("75% :", np.percentile(elevation, 75))
print("99% :", np.percentile(elevation, 99))