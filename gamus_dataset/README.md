# gamus_dataset

A standalone PyTorch `Dataset`/`DataLoader` for fine-tuning Depth Anything V2
on the GAMUS remote-sensing dataset (RGB + nDSM/AGL height).

**This module does not import, wrap, or modify the Depth Anything V2
repository, its model architecture, or its inference code in any way.**
Drop this folder next to (not inside) your DA-V2 checkout and import from
it in a separate training script you write later.

## Status

Dataset loading, preprocessing, and a visual sanity check are implemented.
**No training code exists yet, by design.**

## Known-unverified assumptions

I built this without access to real GAMUS files — only the public Hugging
Face file listing and the GAMUS paper. Before trusting this on a real run,
run `sanity_check.py` and confirm:

| Assumption | Where it's checked |
|---|---|
| Each `<tile>_RGB.h5` holds one `(1024,1024,3)` uint8 array | `dataset._validate_rgb` |
| Each `<tile>_AGL.h5` holds one `(1024,1024)` float32 array (meters) | `dataset._validate_height` |
| Nodata pixels are NaN or in `NODATA_SENTINELS = (-9999, -32768, -1)` | `dataset._build_validity_mask` |
| HDF5 internal dataset key name (unknown — auto-detected) | `dataset._load_h5_array` |

All of these raise a clear, specific error (not a silent wrong answer) if
real data doesn't match — that's intentional.

Also unresolved: the GAMUS paper reports 11,507 total tiles
(6304/1059/4144 train/val/test) but the current Hugging Face mirror lists
5,900 rows (1200/1600/3100). `build_manifest.py` does not try to resolve
this — it just records whatever tiles are physically present in your
downloaded copy.

## Usage

```bash
# 1. Build a reproducible split manifest from your downloaded GAMUS copy
python build_manifest.py --root /path/to/GAMUS --out gamus_manifest.json

# 2. Sanity-check alignment and stats on a few real tiles
python sanity_check.py --root /path/to/GAMUS --manifest gamus_manifest.json --split train --n 3
```

```python
from gamus_dataset import GamusDataset, build_dataloader

train_loader = build_dataloader(
    root="/path/to/GAMUS",
    split="train",
    batch_size=8,
    manifest_path="gamus_manifest.json",
    crop_size=(518, 518),  # matches DINOv2/DA-V2 patch-14 grid
)

for batch in train_loader:
    batch["image"]       # (B, 3, H, W) float, ImageNet-normalized
    batch["target"]      # (B, 1, H, W) float, height in meters (invalid px = 0)
    batch["valid_mask"]  # (B, 1, H, W) bool — use this to mask your loss
    break
```

## Output contract

Every sample is a dict:
- `image`: `(3, H, W)` float tensor, ImageNet-normalized
- `target`: `(1, H, W)` float tensor, raw height in meters, invalid pixels
  zero-filled (do **not** treat 0 as "ground level" without checking
  `valid_mask` first)
- `valid_mask`: `(1, H, W)` bool tensor — `True` where `target` is usable.
  Apply this to your loss (e.g. `loss = (pred - target)[valid_mask]`).
- `tile_id`: `str`, for logging/debugging

## Files

- `dataset.py` — `GamusDataset`, `build_dataloader`
- `transforms.py` — joint resize/crop/flip + ImageNet normalization
- `build_manifest.py` — scans real directories, writes a checksummed split manifest
- `sanity_check.py` — visualizes RGB vs. target + overlay, prints stats, no training
