import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2

# --- ADDED FOR TERRAIN CLASSIFICATION ---
# terrain_classifier.py must sit next to this run.py (or be on PYTHONPATH).
# This import does NOT touch anything related to depth inference - it is
# a fully independent module that only reads raw_image and writes its own
# separate output files.
from terrain_classifier import classify_terrain, make_preview


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')

    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])

    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    # --- ADDED FOR TERRAIN CLASSIFICATION ---
    # Default is ON so terrain_mask.npy is produced automatically. Pass
    # --skip-terrain if you (or your friend) ever want to run pure depth
    # inference with zero extra files written - depth.npy behavior is
    # identical either way.
    parser.add_argument('--skip-terrain', dest='skip_terrain', action='store_true',
                         help='do not run terrain classification, only depth inference')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')

        raw_image = cv2.imread(filename)

        # --- FIX: no tiling. Single inference call keeps depth values
        # consistent across the whole image (relative-depth models are
        # only self-consistent *within* one inference call, not across
        # separately-run tiles). infer_image() internally resizes to the
        # model's working resolution and back, so this works fine up to
        # fairly large images (well beyond our 1024x1024 test tiles).
        depth = depth_anything.infer_image(raw_image, args.input_size)

        # Normalize once, globally, to 0-1
        depth = (depth - depth.min()) / (depth.max() - depth.min())

        # Save raw depth values
        base_name = os.path.splitext(os.path.basename(filename))[0]
        np.save(
            os.path.join(args.outdir, f'{base_name}_depth.npy'),
            depth
        )

        # --- ADDED FOR TERRAIN CLASSIFICATION ---
        # WHY here, and WHY raw_image (not depth_vis or a resized copy):
        # depth_anything.infer_image() already resized its output back to
        # raw_image's original H,W before we saved depth.npy above. So the
        # ONLY way to guarantee terrain_mask.npy lines up pixel-for-pixel
        # with depth.npy is to classify the exact same raw_image array,
        # at its native resolution, with zero extra resizing. Nothing
        # above this block is modified - depth.npy is saved first and is
        # byte-for-byte the same as before this change.
        if not args.skip_terrain:
            terrain_mask = classify_terrain(raw_image)

            # Hard safety check: if this ever fails, something upstream
            # changed raw_image's shape between depth inference and here -
            # better to crash loudly than silently ship a misaligned mask.
            assert terrain_mask.shape == depth.shape, (
                f"terrain_mask shape {terrain_mask.shape} != depth shape "
                f"{depth.shape} for {filename} - alignment broken."
            )

            np.save(
                os.path.join(args.outdir, f'{base_name}_terrain_mask.npy'),
                terrain_mask
            )

            terrain_preview = make_preview(terrain_mask)
            cv2.imwrite(
                os.path.join(args.outdir, f'{base_name}_terrain_mask_preview.png'),
                terrain_preview
            )

        # Create visualization
        depth_vis = (depth * 255.0).astype(np.uint8)

        if args.grayscale:
            depth_vis = np.repeat(depth_vis[..., np.newaxis], 3, axis=-1)
        else:
            depth_vis = (cmap(depth_vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, base_name + '.png'), depth_vis)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth_vis])
            cv2.imwrite(os.path.join(args.outdir, base_name + '.png'), combined_result)