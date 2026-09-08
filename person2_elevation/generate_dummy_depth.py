import numpy as np

# Create a fake depth map: 256x256 grid of random values between 0 and 1
dummy_depth = np.random.rand(256, 256).astype(np.float32)

# Save it to a file so we can use it in later scripts
np.save("person2_elevation/dummy_depth.npy", dummy_depth)

print("Dummy depth array created and saved.")
print("Shape:", dummy_depth.shape)
print("Min value:", dummy_depth.min())
print("Max value:", dummy_depth.max())