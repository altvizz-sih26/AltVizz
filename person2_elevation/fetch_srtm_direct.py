import requests
import gzip
import shutil

tile_name = "N24W078"
url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/N24/{tile_name}.hgt.gz"

print("Downloading:", url)
response = requests.get(url)
response.raise_for_status()  # will error clearly if the download failed

# Save the compressed file
with open("person2_elevation/srtm_tile.hgt.gz", "wb") as f:
    f.write(response.content)

# Decompress it
with gzip.open("person2_elevation/srtm_tile.hgt.gz", "rb") as f_in:
    with open("person2_elevation/srtm_tile.hgt", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

print("SRTM tile downloaded and decompressed: srtm_tile.hgt")