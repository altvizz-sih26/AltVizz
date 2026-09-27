import rasterio
from rasterio.merge import merge
import glob

def mosaic_srtm_tiles(tile_dir="person2_elevation/srtm_tiles", output_path="person2_elevation/mosaic_elevation.tif"):
    tile_paths = glob.glob(f"{tile_dir}/*.hgt")
    
    if not tile_paths:
        raise FileNotFoundError("No .hgt tiles found. Run download_all_tiles.py first.")
    
    print(f"Merging {len(tile_paths)} tiles: {tile_paths}")
    
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic_array, mosaic_transform = merge(datasets)
    
    out_meta = datasets[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic_array.shape[1],
        "width": mosaic_array.shape[2],
        "transform": mosaic_transform,
    })
    
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic_array)
    
    for d in datasets:
        d.close()
    
    print(f"Saved mosaic: {output_path}")
    print(f"Mosaic shape: {mosaic_array.shape}")
    return output_path


if __name__ == "__main__":
    mosaic_srtm_tiles()