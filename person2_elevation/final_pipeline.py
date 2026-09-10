import numpy as np
import rasterio
import matplotlib.pyplot as plt
from depth_loader import load_and_validate_depth  # from earlier file

def load_mosaic_elevation(path="person2_elevation/mosaic_elevation.tif"):
    with rasterio.open(path) as dataset:
        elevation = dataset.read(1).astype(np.float32)
    void_mask = (elevation == -32768) | (elevation < -1000)
    return elevation, void_mask


def calibrate_depth_to_elevation(depth, reference_elevation, void_mask):
    valid = ~void_mask
    a, b = np.polyfit(depth[valid].flatten(), reference_elevation[valid].flatten(), 1)
    calibrated = a * depth + b
    return calibrated, a, b


if __name__ == "__main__":
    # Load Person 1's real depth data (auto-corrected if out of 0-1 range)
    depth = load_and_validate_depth("person2_elevation/person1_depth.npy")
    
    # Load mosaicked real SRTM ground truth
    reference_elevation, void_mask = load_mosaic_elevation()
    
    # NOTE: depth and reference_elevation must be the SAME shape to calibrate directly.
    # If shapes differ (very likely, since they come from different sources/resolutions),
    # reference_elevation must be resampled to match depth's shape first.
    if depth.shape != reference_elevation.shape:
        print(f"WARNING: shape mismatch — depth {depth.shape} vs reference {reference_elevation.shape}.")
        print("Resampling reference elevation to match depth shape (nearest-neighbor).")
        from scipy.ndimage import zoom
        zoom_factors = (depth.shape[0] / reference_elevation.shape[0],
                         depth.shape[1] / reference_elevation.shape[1])
        reference_elevation = zoom(reference_elevation, zoom_factors, order=0)
        void_mask = zoom(void_mask.astype(np.uint8), zoom_factors, order=0).astype(bool)
    
    calibrated_elevation, a, b = calibrate_depth_to_elevation(depth, reference_elevation, void_mask)
    
    valid = ~void_mask
    error = np.abs(calibrated_elevation[valid] - reference_elevation[valid])
    
    print(f"Calibration fit: elevation = {a:.2f} * depth + {b:.2f}")
    print(f"Mean absolute error: {error.mean():.2f} meters")
    print(f"Max absolute error: {error.max():.2f} meters")
    
    np.save("person2_elevation/final_elevation.npy", calibrated_elevation)
    
    plt.figure(figsize=(6,6))
    plt.imshow(calibrated_elevation, cmap='terrain')
    plt.colorbar(label='Elevation (meters)')
    plt.title(f'Final Calibrated Elevation (Mean Error: {error.mean():.1f}m)')
    plt.savefig("person2_elevation/final_elevation_preview.png", dpi=150)
    
    print("\nSaved: final_elevation.npy, final_elevation_preview.png")
    print("Ready for handoff to Person 3.")