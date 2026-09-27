from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "earthflow/GAMUS"
LOCAL_DIR = "./GAMUS"

N_TRAIN = 300
N_VAL = 50

def download_split(api, split: str, n: int):
    rgb_files = sorted(
        f for f in all_files
        if f.startswith(f"images/{split}/") and f.endswith("_RGB.h5")
    )
    subset = rgb_files[:n]
    print(f"[{split}] found {len(rgb_files)} total RGB tiles, downloading {len(subset)}")

    downloaded, missing = 0, []
    for rgb_path in subset:
        tile_id = rgb_path.split("/")[-1][: -len("_RGB.h5")]
        agl_path = f"heights/{split}/{tile_id}_AGL.h5"

        if agl_path not in all_files:
            missing.append(tile_id)
            continue

        hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                         filename=rgb_path, local_dir=LOCAL_DIR)
        hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                         filename=agl_path, local_dir=LOCAL_DIR)
        downloaded += 1

    print(f"[{split}] downloaded {downloaded} pairs, skipped {len(missing)} (no AGL match)")
    if missing:
        print(f"[{split}] example missing tiles: {missing[:5]}")


if __name__ == "__main__":
    api = HfApi()
    print("Listing repo files (one-time, may take a moment for 80GB dataset metadata)...")
    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    print(f"Repo has {len(all_files)} files total.")

    download_split(api, "train", N_TRAIN)
    download_split(api, "val", N_VAL)

    print(f"\nDone. Data is under {Path(LOCAL_DIR).resolve()}")