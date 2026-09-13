import requests


# name: (lon_min, lat_min, lon_max, lat_max)
tiles = {
    "urban": (-90.9229, 30.1831, -90.8789, 30.2211),
    "vegetation": (-75.2783, 40.7140, -75.2344, 40.7473),
    "bare_terrain": (144.9316, -37.5446, 144.9756, -37.5097),
}

for name, (lon_min, lat_min, lon_max, lat_max) in tiles.items():
    print(f"\nDownloading SRTM for {name}...")
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": "SRTMGL1",       # SRTM 30m
        "south": lat_min,
        "north": lat_max,
        "west": lon_min,
        "east": lon_max,
        "outputFormat": "GTiff",
        "API_Key": API_KEY,
    }
    r = requests.get(url, params=params)
    if r.status_code == 200 and len(r.content) > 1000:
        out_path = f"test_data/{name}/srtm_{name}.tif"
        with open(out_path, "wb") as f:
            f.write(r.content)
        print(f"Saved: {out_path} ({len(r.content)} bytes)")
    else:
        print(f"ERROR for {name}: status={r.status_code}")
        print(r.text[:500])
