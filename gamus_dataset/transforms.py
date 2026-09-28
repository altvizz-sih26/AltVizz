"""
Joint geometric transforms applied IDENTICALLY to (image, target, mask), so
spatial correspondence between RGB and height/depth is never broken.

Kept separate from dataset.py (no torch.utils.data imports here) so it can
be reused or unit-tested independently.
"""

import random
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

# ImageNet stats — matches Depth Anything V2's own input preprocessing.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """image: (3, H, W) float in [0, 1] -> ImageNet-normalized."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype).view(3, 1, 1)
    return (image - mean) / std


def joint_geometric_transform(
    image: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    crop_size: Sequence[int],
    train: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Resize then crop image/target/mask with identical geometry.

    Rationale:
      - Image and target are resized with BILINEAR (smooth signals).
      - The mask is resized with NEAREST, so validity is never blended
        across a boundary (a pixel is either valid or it isn't).
      - Invalid target pixels are expected to already be zero-filled
        (done in GamusDataset before this is called), so bilinear
        resizing near an invalid region can leak a small amount of
        "0" into nearby valid pixels; the independently nearest-resized
        mask still correctly flags that thin border band as invalid.
        This is a documented approximation, not a bug.
      - Train: resize so the short side >= crop_size, random-crop to
        crop_size, then a 50% chance horizontal flip (all three tensors
        transformed together, same random draw).
      - Val/Test: resize so the short side == crop_size, center-crop,
        no flip. Fully deterministic.
    """
    ch, cw = crop_size
    _, h, w = image.shape

    scale = max(ch / h, cw / w)
    new_h, new_w = max(ch, round(h * scale)), max(cw, round(w * scale))

    image = F.interpolate(
        image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)
    target = F.interpolate(
        target.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)
    mask = F.interpolate(
        mask.float().unsqueeze(0), size=(new_h, new_w), mode="nearest"
    ).squeeze(0).bool()

    if train:
        top = random.randint(0, new_h - ch)
        left = random.randint(0, new_w - cw)
    else:
        top = (new_h - ch) // 2
        left = (new_w - cw) // 2

    image = image[:, top:top + ch, left:left + cw]
    target = target[:, top:top + ch, left:left + cw]
    mask = mask[:, top:top + ch, left:left + cw]

    if train and ch == cw:
        # Nadir (straight-down) imagery has no fixed "up" direction the way
        # perspective camera depth does (no horizon/gravity cue), so unlike
        # typical monocular depth augmentation, arbitrary 90-degree
        # rotations are geometrically valid here and add useful diversity.
        # Only applied when the crop is square, since rotating a non-square
        # crop 90/270 degrees would swap H and W.
        k = random.randint(0, 3)
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
            target = torch.rot90(target, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])

    if train and random.random() < 0.5:
        image = torch.flip(image, dims=[2])
        target = torch.flip(target, dims=[2])
        mask = torch.flip(mask, dims=[2])

    return image, target, mask
