"""
Overfit sanity test — NOT a real training run.

Purpose: confirm the model/loss/optimizer wiring actually works by
checking it can drive loss to near-zero memorizing just 2 real tiles.
If loss doesn't drop sharply within a few hundred steps, something in the
pipeline is broken (wrong mask usage, gradients not reaching the layer you
think they are, a shape mismatch, learning rate wildly off, etc.) — find
that here, on 2 tiles in seconds, before any real multi-hour run.

This reuses the REAL, unmodified GamusDataset by building a tiny temp
directory of symlinks (images/train/, heights/train/) pointing at
whichever 2 (or more) tiles you give it — so this test exercises the
actual production data-loading code path, not a reimplementation of it.

Usage:
    python overfit_test.py \\
        --tiles-dir /mnt/user-data/uploads \\
        --tile-ids DC_03_26 DC_05_28 \\
        --da2-checkpoint /path/to/depth_anything_v2_vits.pth \\
        --encoder vits \\
        --steps 300
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for gamus_dataset

from gamus_dataset import GamusDataset  # noqa: E402
from gamus_finetune.losses import CombinedHeightLoss  # noqa: E402
from gamus_finetune.model import build_gamus_model  # noqa: E402


def build_toy_gamus_root(tiles_dir: Path, tile_ids: list, workdir: Path) -> Path:
    """
    Symlinks the given tiles' RGB/AGL files into workdir/images/train/ and
    workdir/heights/train/, so the real GamusDataset (which expects that
    directory layout) can be used unmodified on just these tiles.
    """
    img_dir = workdir / "images" / "train"
    agl_dir = workdir / "heights" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    agl_dir.mkdir(parents=True, exist_ok=True)

    for tile_id in tile_ids:
        rgb_src = tiles_dir / "images" / "train" / f"{tile_id}_RGB.h5"
        agl_src = tiles_dir / "heights" / "train" / f"{tile_id}_AGL.h5"
        if not rgb_src.exists() or not agl_src.exists():
            raise FileNotFoundError(f"Missing RGB/AGL pair for tile {tile_id} in {tiles_dir}")
        (img_dir / rgb_src.name).symlink_to(rgb_src.resolve())
        (agl_dir / agl_src.name).symlink_to(agl_src.resolve())

    return workdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", required=True, help="folder containing the sample *_RGB.h5/*_AGL.h5")
    ap.add_argument("--tile-ids", nargs="+", required=True, help="e.g. DC_03_26 DC_05_28")
    ap.add_argument("--da2-checkpoint", required=True)
    ap.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--crop-size", type=int, default=518)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()
    assert args.crop_size % 14 == 0, (
        f"--crop-size must be a multiple of 14 (DINOv2 patch size), got {args.crop_size}"
    )

    tiles_dir = Path(args.tiles_dir)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = build_toy_gamus_root(tiles_dir, args.tile_ids, Path(tmp))

        dataset = GamusDataset(
            root=str(workdir),
            split="train",
            crop_size=(args.crop_size, args.crop_size),
            augment=False,  # deterministic — we WANT the model to see identical crops each step
        )
        print(f"Overfit test on {len(dataset)} tile(s): {args.tile_ids}")

        # Load the full tiny set once, keep it resident (no need to
        # re-run the dataloader/workers machinery for 2 samples).
        samples = [dataset[i] for i in range(len(dataset))]
        images = torch.stack([s["image"] for s in samples]).to(args.device)
        targets = torch.stack([s["target"] for s in samples]).to(args.device)
        masks = torch.stack([s["valid_mask"] for s in samples]).to(args.device)

        model = build_gamus_model(args.encoder, args.da2_checkpoint, device=args.device)
        # Deliberately do NOT freeze the backbone here: for a memorization
        # test we want maximum capacity to drive loss to zero fast. The
        # freeze schedule matters for real training generalization, not
        # for this wiring check.
        loss_fn = CombinedHeightLoss()
        optimizer = AdamW(model.parameters(), lr=args.lr)

        print("Initial check: forward pass shape ->", end=" ")
        with torch.no_grad():
            pred0 = model(images)
        print(tuple(pred0.shape), "(expected: matches target shape", tuple(targets.shape), ")")
        assert pred0.shape == targets.shape, (
            "Model output shape doesn't match target shape — fix this before "
            "trusting anything else in this test."
        )

        losses = []
        for step in range(args.steps):
            pred = model(images)
            loss, parts = loss_fn(pred, targets, masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if step % args.log_every == 0 or step == args.steps - 1:
                print(f"step {step:4d}  loss {loss.item():.4f}  "
                      f"(l1={parts['l1']:.4f} grad={parts['grad']:.4f})")

        print("\n--- Result ---")
        print(f"First-step loss:  {losses[0]:.4f}")
        print(f"Final-step loss:  {losses[-1]:.4f}")
        if losses[-1] < 0.1 * losses[0]:
            print("PASS-ish: loss dropped sharply — wiring looks correct. "
                  "This does NOT mean the model is good, only that gradients "
                  "flow and the loss is actually being minimized.")
        else:
            print("WARNING: loss did not drop much. Something in the "
                  "model/loss/data wiring is likely broken — check that "
                  "gradients reach the parameters you expect "
                  "(print .grad on a few), that the mask isn't zeroing out "
                  "everything, and that pred/target aren't secretly on "
                  "different scales.")


if __name__ == "__main__":
    main()
