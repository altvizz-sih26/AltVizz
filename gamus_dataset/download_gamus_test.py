import random
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "earthflow/GAMUS"
LOCAL_DIR = "/Users/tanishka/Desktop/AltVizz/gamus_dataset/GAMUS_test"

N_TEST = 300
SEED = 0

api = HfApi()

print("Listing repo files...")

all_files = set(
    api.list_repo_files(
        repo_id=REPO_ID,
        repo_type="dataset"
    )
)

rgb_files = sorted(
    f for f in all_files
    if f.startswith("images/test/") and f.endswith("_RGB.h5")
)

print(f"[test] found {len(rgb_files)} RGB tiles")

subset = sorted(
    random.Random(SEED).sample(
        rgb_files,
        min(N_TEST, len(rgb_files))
    )
)

downloaded = 0
missing = []

for i, rgb_path in enumerate(subset, 1):
    tile_id = rgb_path.split("/")[-1][:-len("_RGB.h5")]
    agl_path = f"heights/test/{tile_id}_AGL.h5"

    if agl_path not in all_files:
        missing.append(tile_id)
        continue

    hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=rgb_path,
        local_dir=LOCAL_DIR
    )

    hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=agl_path,
        local_dir=LOCAL_DIR
    )

    downloaded += 1

    if downloaded % 25 == 0:
        print(f"[test] downloaded {downloaded}/{N_TEST} pairs")

print()
print(f"[test] selected: {len(subset)}")
print(f"[test] downloaded pairs: {downloaded}")
print(f"[test] missing AGL: {len(missing)}")

if missing:
    print("Missing AGL:")
    for x in missing:
        print(x)

print(f"Data is under: {Path(LOCAL_DIR).resolve()}")