from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path
import shutil

REPO_ID = "earthflow/GAMUS"
N = 50

OUTPUT = Path("gamus_test")

for folder in ["images", "heights", "classes"]:
    (OUTPUT / folder).mkdir(parents=True, exist_ok=True)

api = HfApi()

print("Finding GAMUS test files...")

files = api.list_repo_tree(
    REPO_ID,
    repo_type="dataset",
    recursive=True
)

paths = [f.path for f in files]

images = {
    Path(p).stem.replace("_RGB", ""): p
    for p in paths
    if p.startswith("images/test/")
}

heights = {
    Path(p).stem.replace("_AGL", ""): p
    for p in paths
    if p.startswith("heights/test/")
}

classes = {
    Path(p).stem.replace("_CLS", ""): p
    for p in paths
    if p.startswith("classes/test/")
}

sample_ids = sorted(set(images) & set(heights) & set(classes))[:N]

print(f"Found {len(sample_ids)} complete samples.")
print("Downloading...")

for i, sample_id in enumerate(sample_ids, 1):

    for folder, repo_path in [
        ("images", images[sample_id]),
        ("heights", heights[sample_id]),
        ("classes", classes[sample_id])
    ]:

        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=repo_path
        )

        shutil.copy2(
            downloaded,
            OUTPUT / folder / Path(repo_path).name
        )

    print(f"{i}/{len(sample_ids)}  {sample_id}")

print("\nDONE")
print(f"Saved to: {OUTPUT.resolve()}")