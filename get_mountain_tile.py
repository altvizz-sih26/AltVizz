"""
Download a 1024x1024 true-colour Sentinel-2 tile (10 m/pixel, ~10 km square) around any
latitude/longitude and save it as an 8-bit RGB GeoTIFF that DepthWizard can use directly.

Setup (once):   pip install pystac-client planetary-computer rasterio numpy
Run:            python get_mountain_tile.py LAT LON output.tif
Examples:       python get_mountain_tile.py 27.9881 86.9250 everest.tif      (Mount Everest)
                python get_mountain_tile.py 28.5300 83.8800 annapurna.tif    (Annapurna, Nepal)
                python get_mountain_tile.py 45.9763 7.6586 matterhorn.tif    (Swiss Alps)
Needs internet. Uses the free Microsoft Planetary Computer; no account or key is required.
"""
import sys
import rasterio
import planetary_computer as pc
from pystac_client import Client
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

SIZE = 1024  # pixels per side (10 m each -> about 10.24 km)

if len(sys.argv) != 4:
    sys.exit(__doc__)
lat, lon, out_path = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects={"type": "Point", "coordinates": [lon, lat]},
    datetime="2022-01-01/2025-12-31",
    query={"eo:cloud_cover": {"lt": 15}},
    sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    max_items=1,
)
items = list(search.items())
if not items:
    sys.exit("No low-cloud Sentinel-2 scene found here. Try another place, or raise the cloud limit (15) in the script.")
item = items[0]
print("Using scene:", item.id, "| cloud cover:", item.properties.get("eo:cloud_cover"), "| date:", item.datetime)

with rasterio.open(item.assets["visual"].href) as src:   # 'visual' = ready-made 8-bit true colour
    xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
    row, col = src.index(xs[0], ys[0])
    window = Window(col - SIZE // 2, row - SIZE // 2, SIZE, SIZE)
    data = src.read(window=window, boundless=True, fill_value=0)
    profile = {
        "driver": "GTiff", "height": SIZE, "width": SIZE, "count": 3, "dtype": "uint8",
        "crs": src.crs, "transform": src.window_transform(window),
        "compress": "deflate", "photometric": "RGB",
    }
with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(data)
print("Saved", out_path, "-", data.shape[2], "x", data.shape[1], "pixels, CRS", profile["crs"])
