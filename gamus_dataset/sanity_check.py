"""
Visual + numeric sanity check for the GAMUS dataset wrapper.

Confirms, for a few tiles:
  - RGB and height load and pair up correctly
  - RGB and height are spatially aligned (side-by-side + overlay)
  - per-sample stats: shape, dtype, % valid pixels, target min/max/mean

This does NOT train anything. Run it before writing any training code,
and re-run it once you have real GAMUS files downloaded — this module
currently assumes an HDF5 layout inferred from file-size math on the
Hugging Face listing (see the earlier findings report), and this script
is how you confirm or correct those assumptions against real data.

Usage:
    python sanity_check.py --root /path/to/GAMUS --split train --n 3
    python sanity_check.py --root /path/to/GAMUS --manifest gamus_manifest.json --split val
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# This file lives inside gamus_dataset/ itself, importing its own package
# by absolute name (`from gamus_dataset...`) — that only resolves if the
# directory ABOVE gamus_dataset/ is on sys.path. Add it explicitly so this
# works regardless of how the script is invoked (matches the same pattern
# used in gamus_finetune/train.py and overfit_test.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gamus_dataset.dataset import GamusDataset
from gamus_dataset.transforms import IMAGENET_MEAN, IMAGENET_STD


def denormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = image * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--crop", type=int, default=518)
    ap.add_argument("--out", default="gamus_sanity_check.png")
    args = ap.parse_args()

    dataset = GamusDataset(
        root=args.root,
        split=args.split,
        manifest_path=args.manifest,
        crop_size=(args.crop, args.crop),
        augment=False,  # deterministic center-crop for inspection
    )
    print(f"Dataset size: {len(dataset)}")

    n = min(args.n, len(dataset))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        sample = dataset[i]
        image, target, mask = sample["image"], sample["target"], sample["valid_mask"]
        tile_id = sample["tile_id"]

        target_np = target.squeeze(0).numpy()
        mask_np = mask.squeeze(0).numpy()
        valid_vals = target_np[mask_np]

        print(f"\n[{tile_id}]")
        print(f"  image:  shape={tuple(image.shape)} dtype={image.dtype}")
        print(f"  target: shape={tuple(target.shape)} dtype={target.dtype}")
        print(f"  valid pixels: {mask_np.mean() * 100:.2f}%")
        if valid_vals.size > 0:
            print(
                f"  target (valid only): min={valid_vals.min():.3f} "
                f"max={valid_vals.max():.3f} mean={valid_vals.mean():.3f}"
            )
        else:
            print("  WARNING: no valid target pixels in this crop.")

        img_vis = denormalize(image)

        axes[i, 0].imshow(img_vis)
        axes[i, 0].set_title(f"{tile_id} — RGB")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(target_np, cmap="viridis")
        axes[i, 1].set_title("Height / target")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img_vis)
        axes[i, 2].imshow(
            np.where(mask_np, target_np, np.nan), cmap="viridis", alpha=0.5
        )
        axes[i, 2].set_title("Overlay (alignment check)")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved visualization to {args.out}")
    print(
        "Look for: does high ground (buildings, trees) in the height map "
        "line up with the same objects in the RGB image? If it's offset "
        "or mirrored, RGB/height are NOT spatially aligned as assumed."
    )


if __name__ == "__main__":
    main()