import numpy as np

def relative_depth_to_elevation(depth_array, min_height=0, max_height=50, invert=False):
    """
    Converts a 0-1 relative depth array into a fake elevation surface.
    
    depth_array: the raw depth values (0 to 1)
    min_height, max_height: the height range we want in meters (arbitrary for now)
    invert: if True, flips the values before converting.
            Not needed here since Person 1's convention already matches what we want:
            0 = far from camera = low elevation
            1 = close to camera = high elevation
    """
    if invert:
        depth_array = 1 - depth_array  # flip the values, only used if needed later
    
    # Stretch values from 0-1 range into min_height-max_height range
    elevation = depth_array * (max_height - min_height) + min_height
    
    return elevation.astype(np.float32)


if __name__ == "__main__":
    # Load the dummy depth file we created earlier
    dummy_depth = np.load("person2_elevation/dummy_depth.npy")
    
    # Convert it
    elevation = relative_depth_to_elevation(dummy_depth)
    
    # Save the result
    np.save("person2_elevation/relative_elevation.npy", elevation)
    
    print("Elevation surface created.")
    print("Shape:", elevation.shape)
    print("Min height:", elevation.min())
    print("Max height:", elevation.max())