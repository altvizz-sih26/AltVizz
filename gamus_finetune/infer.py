"""
Inference with the fine-tuned GAMUS height model — separate from, and not
a replacement for, DA-V2's own run.py. For each image it saves:

    <name>_height_meters.npy      raw metric height in meters (can be negative)
    <name>_height_normalized.npy  float32, min-max to [0, 1], matches the
                                  depth_loader.py format contract
    <name>_terrain_mask.npy       uint8 class IDs 0..5 from the project's own
                                  terrain_classifier (urban, vegetation,
                                  bare_terrain, water, shadow, unknown);
                                  skip with --skip-terrain

Pass --visualize to also save a colorized RGB+height preview PNG, and a
terrain mask preview PNG (both off by default).

The terrain mask comes from terrain_classifier.classify_terrain, imported
unmodified and run on the same raw image at its original resolution, so it
lines up pixel-for-pixel with the height arrays. Class IDs are passed
through exactly as the classifier produces them (no remapping).

Reuses DA-V2's own, unmodified `image2tensor` preprocessing (the same
resize-to-multiple-of-14 + ImageNet normalize that run.py's infer_image()
uses internally), then runs the FIXED forward pass from gamus_finetune
(no double ReLU floor — see model.py), so output can represent the small
negative heights real GAMUS data has near ground level.

Usage:
    # height + terrain mask (default)
    python3 infer.py \\
        --img-path /path/to/image_or_folder \\
        --checkpoint runs/gamus_vits_run1/best.pt \\
        --da2-checkpoint /path/to/depth_anything_v2_vits.pth \\
        --encoder vits

    # plus preview PNGs
    python infer.py ... --visualize

    # height only, no terrain mask
    python infer.py ... --skip-terrain
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for gamus_dataset (not strictly needed here, kept for consistency)

from gamus_finetune.model import build_gamus_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="GAMUS fine-tuned height inference")
    ap.add_argument("--img-path", required=True,
                     help="image file, folder, or .txt file list (same conventions as DA-V2's run.py)")
    ap.add_argument("--checkpoint", required=True, help="fine-tuned checkpoint, e.g. best.pt")
    ap.add_argument("--da2-checkpoint", required=True,
                     help="original pretrained DA-V2 .pth, needed to build the architecture "
                          "before loading fine-tuned weights on top")
    ap.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    ap.add_argument("--outdir", default="./height_predictions")
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--visualize", action="store_true",
                     help="also save RGB+height and terrain mask preview PNGs (off by default)")
    ap.add_argument("--skip-terrain", action="store_true",
                     help="do not run terrain classification, save height arrays only")
    ap.add_argument("--terrain-dir", default=str(Path(__file__).resolve().parent.parent / "terrain_classifier"),
                     help="folder containing terrain_classifier.py "
                          "(default: <repo root>/terrain_classifier)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    assert args.input_size % 14 == 0, f"--input-size must be a multiple of 14, got {args.input_size}"

    classify_terrain = make_preview = None
    if not args.skip_terrain:
        terrain_file = Path(args.terrain_dir) / "terrain_classifier.py"
        if not terrain_file.exists():
            raise FileNotFoundError(
                f"{terrain_file} not found. Pass --terrain-dir pointing at the folder "
                f"that contains terrain_classifier.py, or use --skip-terrain."
            )
        # Put the classifier's own folder FIRST on the path, so that
        # `terrain_classifier` resolves to terrain_classifier.py and not to
        # the folder of the same name at the repo root.
        sys.path.insert(0, str(Path(args.terrain_dir).resolve()))
        from terrain_classifier import classify_terrain, make_preview  # noqa: E402

    os.makedirs(args.outdir, exist_ok=True)

    model = build_gamus_model(args.encoder, args.da2_checkpoint, device=args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded fine-tuned checkpoint (epoch {ckpt.get('epoch', '?')}, "
          f"val_metrics at save time: {ckpt.get('val_metrics', {})})")

    if os.path.isfile(args.img_path):
        if args.img_path.endswith("txt"):
            with open(args.img_path) as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = [f for f in glob.glob(os.path.join(args.img_path, "**/*"), recursive=True)
                     if os.path.isfile(f)]

    cmap = matplotlib.colormaps.get_cmap("Spectral_r") if args.visualize else None

    for k, filename in enumerate(filenames):
        print(f"Progress {k + 1}/{len(filenames)}: {filename}")
        raw_image = cv2.imread(filename)
        if raw_image is None:
            print("  skipped (could not read as an image)")
            continue

        # DA-V2's own unmodified preprocessing utility (resize to a
        # multiple of 14, ImageNet normalize) — the same one run.py's
        # infer_image() uses internally, kept identical for consistency.
        image_tensor, (h, w) = model.backbone_and_head.image2tensor(raw_image, args.input_size)
        # image2tensor auto-picks its own device internally; force it back
        # to whatever device the model actually lives on, in case they differ.
        image_tensor = image_tensor.to(args.device)

        with torch.no_grad():
            height_pred = model(image_tensor)  # (1, 1, ph*14, pw*14) meters, can be negative
            height_pred = F.interpolate(height_pred, (h, w), mode="bilinear", align_corners=True)

        height_np = height_pred[0, 0].cpu().numpy()

        base_name = os.path.splitext(os.path.basename(filename))[0]

        # Raw metric height, in meters (can be negative) — for our own
        # inspection/documentation, where "3.2 meters" is meaningful.
        np.save(os.path.join(args.outdir, f"{base_name}_height_meters.npy"), height_np)

        # Normalized to [0, 1], float32 — matches Person 2's format
        # contract (depth_loader.py: float32, 0-1, 0=far/1=close), same
        # min-max convention DA-V2's own run.py uses. This is the file to
        # hand off downstream; the _meters one above is for us.
        h_min, h_max = height_np.min(), height_np.max()
        height_normalized = ((height_np - h_min) / (h_max - h_min + 1e-8)).astype(np.float32)
        np.save(os.path.join(args.outdir, f"{base_name}_height_normalized.npy"), height_normalized)

        # Terrain mask from the project's own classifier, run on the SAME
        # raw_image at its native resolution (the height array was resized
        # back to exactly this H,W above), so the two align pixel-for-pixel.
        # Class IDs pass through unchanged: 0 urban, 1 vegetation,
        # 2 bare_terrain, 3 water, 4 shadow, 5 unknown_low_confidence.
        if not args.skip_terrain:
            terrain_mask = classify_terrain(raw_image)

            # Fail loudly rather than ship a misaligned mask.
            assert terrain_mask.shape == height_np.shape, (
                f"terrain_mask shape {terrain_mask.shape} != height shape "
                f"{height_np.shape} for {filename} - alignment broken."
            )

            np.save(os.path.join(args.outdir, f"{base_name}_terrain_mask.npy"), terrain_mask)

            if args.visualize:
                cv2.imwrite(os.path.join(args.outdir, f"{base_name}_terrain_mask_preview.png"),
                            make_preview(terrain_mask))

        if args.visualize:
            vis = (height_np - height_np.min()) / (height_np.max() - height_np.min() + 1e-8)
            vis_img = (cmap(vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined = cv2.hconcat([raw_image, split_region, vis_img])
            cv2.imwrite(os.path.join(args.outdir, f"{base_name}_height_preview.png"), combined)

    print(f"\nDone. Height predictions saved to {args.outdir}")


if __name__ == "__main__":
    main()