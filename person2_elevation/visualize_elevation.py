import numpy as np
import matplotlib.pyplot as plt

# Load the elevation data we generated earlier
elevation = np.load("person2_elevation/relative_elevation.npy")

# Create the plot
plt.figure(figsize=(6, 6))
plt.imshow(elevation, cmap='terrain')
plt.colorbar(label='Elevation (meters)')
plt.title('Relative Elevation Map (dummy data)')

# Save it as an image file
plt.savefig("person2_elevation/relative_elevation_preview.png", dpi=150)

print("Visualization saved as relative_elevation_preview.png")