import h5py
from pathlib import Path
from PIL import Image

SRC_DIR = Path("gamus_test/images")
OUT_DIR = Path("gamus_test/rgb_png")

OUT_DIR.mkdir(parents=True, exist_ok=True)

h5_files = sorted(SRC_DIR.glob("*_RGB.h5"))

print(f"Found {len(h5_files)} RGB files to export.")

for i, path in enumerate(h5_files, 1):
    sample_id = path.stem.replace("_RGB", "")

    with h5py.File(path, "r") as f:
        rgb = f["image"][()]

    out_path = OUT_DIR / f"{sample_id}.png"
    Image.fromarray(rgb).save(out_path)

    print(f"{i}/{len(h5_files)}  {sample_id} -> {out_path.name}")

print("\nDONE")
print(f"Saved {len(h5_files)} PNGs to: {OUT_DIR.resolve()}")