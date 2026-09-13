import numpy as np

def load_and_validate_depth(path):
    """
    Loads a depth .npy file and validates/fixes it against our format contract:
    float32, normalized 0-1, 0=far, 1=close.
    Auto-normalizes if out of range, with a warning.
    """
    depth = np.load(path)
    
    if depth.dtype != np.float32:
        print(f"WARNING: expected float32, got {depth.dtype}. Converting.")
        depth = depth.astype(np.float32)
    
    d_min, d_max = depth.min(), depth.max()
    
    if d_min < 0 or d_max > 1:
        print(f"WARNING: depth values out of 0-1 range (min={d_min:.3f}, max={d_max:.3f}).")
        print("Auto-normalizing to 0-1. Flag this file to Person 1 as non-compliant with format contract.")
        depth = (depth - d_min) / (d_max - d_min)
    
    return depth.astype(np.float32)


if __name__ == "__main__":
    depth = load_and_validate_depth("person2_elevation/person1_depth.npy")
    print("Loaded shape:", depth.shape)
    print("Final min/max:", depth.min(), depth.max())