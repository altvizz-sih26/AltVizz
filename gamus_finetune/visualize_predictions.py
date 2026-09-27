"""
Loads a trained checkpoint (e.g. best.pt from train.py) and visualizes its
predictions against ground truth on a few real tiles — RGB, ground-truth
height, predicted height (same color scale, for a fair visual comparison),
and an error map, plus the same per-bucket metrics train.py reports.

This is for LOOKING at what the model actually does, which usually reveals
things aggregate metrics hide (e.g. "is it adding noise to flat ground",
"does it miss building edges", "is it systematically biased on one class
of terrain") — complements, doesn't replace, the numeric metrics.

Usage:
    python visualize_predictions.py \\
        --gamus-root /path/to/GAMUS \\
        --manifest gamus_manifest.json \\
        --checkpoint runs/gamus_vits_run1/best.pt \\
        --da2-checkpoint /path/to/depth_anything_v2_vits.pth \\
        --encoder vits \\
        --split val --n 4
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for gamus_dataset

from gamus_dataset import GamusDataset  # noqa: E402
from gamus_dataset.transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from gamus_finetune.model import build_gamus_model  # noqa: E402
from gamus_finetune.train import compute_metrics  # noqa: E402  (reuse, don't reimplement)


def denormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = image * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamus-root", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--checkpoint", required=True, help="path to best.pt / latest.pt from train.py")
    ap.add_argument("--da2-checkpoint", required=True,
                     help="original DA-V2 .pth, needed to build the architecture before "
                          "loading fine-tuned weights on top")
    ap.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--crop-size", type=int, default=518)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="gamus_predictions.png")
    args = ap.parse_args()
    assert args.crop_size % 14 == 0, (
        f"--crop-size must be a multiple of 14 (DINOv2 patch size), got {args.crop_size}"
    )

    dataset = GamusDataset(
        root=args.gamus_root,
        split=args.split,
        manifest_path=args.manifest,
        crop_size=(args.crop_size, args.crop_size),
        augment=False,  # deterministic center-crop, so this is reproducible to re-inspect
    )
    print(f"Dataset size: {len(dataset)}")
    n = min(args.n, len(dataset))

    model = build_gamus_model(args.encoder, args.da2_checkpoint, device=args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_metrics at save time: {ckpt.get('val_metrics', {})})")

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        sample = dataset[i]
        image = sample["image"].unsqueeze(0).to(args.device)
        target = sample["target"].unsqueeze(0).to(args.device)
        valid_mask = sample["valid_mask"].unsqueeze(0).to(args.device)
        tile_id = sample["tile_id"]

        with torch.no_grad():
            pred = model(image)

        metrics = compute_metrics(pred, target, valid_mask)
        print(f"\n[{tile_id}] mae={metrics['mae']:.3f} rmse={metrics['rmse']:.3f}")
        for lo_hi in ("0-2m", "2-10m", "10-infm"):
            print(f"  {lo_hi}: mae={metrics.get(f'mae_{lo_hi}', float('nan')):.3f} "
                  f"n={metrics.get(f'n_{lo_hi}', 0)}")

        image_vis = denormalize(sample["image"])
        target_np = target[0, 0].cpu().numpy()
        pred_np = pred[0, 0].cpu().numpy()
        mask_np = valid_mask[0, 0].cpu().numpy()
        error_np = np.abs(pred_np - target_np)

        # Shared color scale between GT and prediction so the comparison
        # is visually meaningful (not two independently-normalized plots).
        valid_vals = np.concatenate([target_np[mask_np], pred_np[mask_np]]) if mask_np.any() else np.array([0.0])
        vmin, vmax = valid_vals.min(), valid_vals.max()

        axes[i, 0].imshow(image_vis)
        axes[i, 0].set_title(f"{tile_id} — RGB")
        axes[i, 0].axis("off")

        im1 = axes[i, 1].imshow(np.where(mask_np, target_np, np.nan), cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title("Ground truth height")
        axes[i, 1].axis("off")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

        im2 = axes[i, 2].imshow(pred_np, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i, 2].set_title("Predicted height")
        axes[i, 2].axis("off")
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

        im3 = axes[i, 3].imshow(np.where(mask_np, error_np, np.nan), cmap="magma")
        axes[i, 3].set_title(f"Abs error (mae={metrics['mae']:.2f}m)")
        axes[i, 3].axis("off")
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved visualization to {args.out}")


if __name__ == "__main__":
    main()
