import requests
import gzip
import shutil
import os

TILE_NAMES = [
    "N23W079", "N24W079", "N25W079",
    "N23W078", "N24W078", "N25W078",
    "N23W077", "N24W077", "N25W077",
]

def download_tile(tile_name, out_dir="person2_elevation/srtm_tiles"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tile_name}.hgt")
    
    if os.path.exists(out_path):
        print(f"{tile_name} already exists, skipping.")
        return out_path
    
    lat_band = tile_name[:3]  # e.g. "N24"
    url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat_band}/{tile_name}.hgt.gz"
    
    print(f"Downloading {tile_name}...")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"  FAILED ({response.status_code}) - likely ocean-only tile, skipping.")
        return None
    
    gz_path = out_path + ".gz"
    with open(gz_path, "wb") as f:
        f.write(response.content)
    
    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    os.remove(gz_path)
    print(f"  Saved {out_path}")
    return out_path


if __name__ == "__main__":
    successful = []
    for tile in TILE_NAMES:
        result = download_tile(tile)
        if result:
            successful.append(result)
    
    print(f"\nDownloaded {len(successful)} of {len(TILE_NAMES)} tiles.")