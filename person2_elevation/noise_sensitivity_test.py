import numpy as np
import rasterio

def load_srtm_elevation(path):
    with rasterio.open(path) as dataset:
        elevation = dataset.read(1).astype(np.float32)
    void_mask = (elevation == -32768) | (elevation < -1000)
    return elevation, void_mask


def generate_synthetic_depth_from_elevation(elevation, void_mask, noise_level):
    e_min = np.nanmin(np.where(void_mask, np.nan, elevation))
    e_max = np.nanmax(np.where(void_mask, np.nan, elevation))
    depth = (elevation - e_min) / (e_max - e_min)
    noise = np.random.normal(0, noise_level, depth.shape)
    depth = np.clip(depth + noise, 0, 1).astype(np.float32)
    return depth


def calibrate_depth_to_elevation(depth, reference_elevation, void_mask):
    valid = ~void_mask
    depth_valid = depth[valid].flatten()
    elevation_valid = reference_elevation[valid].flatten()
    a, b = np.polyfit(depth_valid, elevation_valid, 1)
    calibrated_elevation = a * depth + b
    return calibrated_elevation, a, b


if __name__ == "__main__":
    real_elevation, void_mask = load_srtm_elevation("person2_elevation/N24W078.hgt")
    valid = ~void_mask
    
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    
    print(f"{'Noise Level':>12} | {'Mean Error (m)':>15} | {'Max Error (m)':>15}")
    print("-" * 48)
    
    results = []
    for noise in noise_levels:
        depth = generate_synthetic_depth_from_elevation(real_elevation, void_mask, noise)
        calibrated, a, b = calibrate_depth_to_elevation(depth, real_elevation, void_mask)
        
        error = np.abs(calibrated[valid] - real_elevation[valid])
        mean_err = error.mean()
        max_err = error.max()
        
        results.append((noise, mean_err, max_err))
        print(f"{noise*100:>10.0f}% | {mean_err:>15.2f} | {max_err:>15.2f}")
    
    # Save results as a simple CSV for reference/reporting
    with open("person2_elevation/noise_sensitivity_results.csv", "w") as f:
        f.write("noise_level,mean_error_m,max_error_m\n")
        for noise, mean_err, max_err in results:
            f.write(f"{noise},{mean_err:.2f},{max_err:.2f}\n")
    
    print("\nSaved results to noise_sensitivity_results.csv")