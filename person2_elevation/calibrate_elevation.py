import numpy as np
import rasterio

def load_srtm_elevation(path):
    """Load SRTM data and mask out void/no-data pixels."""
    with rasterio.open(path) as dataset:
        elevation = dataset.read(1).astype(np.float32)
    
    # SRTM uses -32768 as the official "no data" marker
    void_mask = (elevation == -32768)
    
    # Also treat extreme unrealistic values as suspect (safety net)
    void_mask = void_mask | (elevation < -1000)
    
    print(f"Void/no-data pixels: {void_mask.sum()} out of {elevation.size} ({100*void_mask.sum()/elevation.size:.1f}%)")
    
    return elevation, void_mask


def generate_synthetic_depth_from_elevation(elevation, void_mask, noise_level=0.05):
    """
    Create a fake 'depth' array derived from real elevation, to test our
    calibration pipeline against known ground truth before Person 1's real data arrives.
    """
    valid_elevation = np.where(void_mask, np.nan, elevation)
    
    e_min = np.nanmin(valid_elevation)
    e_max = np.nanmax(valid_elevation)
    
    # Normalize elevation to 0-1, matching Person 1's depth convention (high=close=high elevation)
    depth = (elevation - e_min) / (e_max - e_min)
    
    # Add realistic noise, since real depth models are never perfectly accurate
    noise = np.random.normal(0, noise_level, depth.shape)
    depth = np.clip(depth + noise, 0, 1).astype(np.float32)
    
    return depth


def calibrate_depth_to_elevation(depth, reference_elevation, void_mask):
    """
    Core calibration function: fits a linear relationship between depth values
    and real SRTM elevation, using only valid (non-void) pixels.
    """
    valid = ~void_mask
    
    depth_valid = depth[valid].flatten()
    elevation_valid = reference_elevation[valid].flatten()
    
    # Fit: elevation = a * depth + b  (simple linear regression)
    a, b = np.polyfit(depth_valid, elevation_valid, 1)
    
    print(f"Calibration fit: elevation = {a:.2f} * depth + {b:.2f}")
    
    # Apply this calibration to the FULL depth array
    calibrated_elevation = a * depth + b
    
    return calibrated_elevation, a, b


if __name__ == "__main__":
    # Load real SRTM data
    real_elevation, void_mask = load_srtm_elevation("person2_elevation/N24W078.hgt")
    
    # Generate synthetic depth for testing (stand-in for Person 1's real output)
    synthetic_depth = generate_synthetic_depth_from_elevation(real_elevation, void_mask)
    
    # Run calibration
    calibrated_elevation, a, b = calibrate_depth_to_elevation(synthetic_depth, real_elevation, void_mask)
    
    # Check how close we got to the real values (sanity check)
    valid = ~void_mask
    error = np.abs(calibrated_elevation[valid] - real_elevation[valid])
    print(f"Mean absolute error: {error.mean():.2f} meters")
    print(f"Max absolute error: {error.max():.2f} meters")
    
    np.save("person2_elevation/calibrated_elevation.npy", calibrated_elevation)
    print("Saved calibrated_elevation.npy")