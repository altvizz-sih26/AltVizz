"""
GAMUS Dataset/DataLoader for depth (nDSM/height) fine-tuning.

This module is fully self-contained: it does NOT import, wrap, or modify
anything from the Depth Anything V2 repository or its inference code. It
just produces plain torch.Tensor batches of {image, target, valid_mask}
that any training loop (including one built around the unmodified DA-V2
model) can consume.

IMPORTANT — unverified assumptions (see the earlier findings report):
  - Each `<tile>_RGB.h5` file is assumed to hold exactly one HDF5 dataset
    containing a (1024, 1024, 3) uint8 array.
  - Each `<tile>_AGL.h5` file is assumed to hold exactly one HDF5 dataset
    containing a (1024, 1024) float32 array (nDSM height, in meters).
  - Invalid/nodata target pixels are assumed to be NaN and/or one of the
    sentinel values in NODATA_SENTINELS below.

These are checked at runtime (see `_load_h5_array`, `_validate_*`,
`_build_validity_mask`) and will raise an informative error rather than
fail silently if real files don't match. Run `sanity_check.py` on a
handful of real tiles before trusting any of this.
"""

import json
import logging
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import joint_geometric_transform, normalize_image

logger = logging.getLogger(__name__)

# Sentinel "nodata" values seen in comparable nDSM/DSM remote-sensing
# datasets (Potsdam/Vaihingen/DFC19-style). Extend this if sanity_check.py
# reveals a different sentinel in the real GAMUS files.
NODATA_SENTINELS = (-9999.0, -32768.0, -1.0)

# Below this height (meters), treat a pixel as invalid rather than a
# legitimate, slightly-negative, noisy ground reading. Tune after
# inspecting real data.
MIN_VALID_HEIGHT = -1.0

# Common candidate dataset-key names to try if a .h5 file has more than
# one top-level dataset (see _load_h5_array).
_KEY_CANDIDATES = ("data", "image", "rgb", "height", "agl", "dsm", "ndsm")


def _load_h5_array(path: Path) -> np.ndarray:
    """Load the array stored in a GAMUS .h5 tile, auto-detecting the key."""
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if len(keys) == 1:
            return f[keys[0]][()]
        for name in _KEY_CANDIDATES:
            if name in f:
                return f[name][()]
        raise KeyError(
            f"Could not determine which dataset to read in {path}. "
            f"Available keys: {keys}. Add the real key to "
            f"`_KEY_CANDIDATES` in dataset.py."
        )


class GamusDataset(Dataset):
    """
    Loads GAMUS RGB/height tile pairs and returns dicts with:
        image:       (3, H, W) float tensor, ImageNet-normalized
        target:      (1, H, W) float tensor, height in meters
                     (invalid pixels zero-filled — use valid_mask, not
                     this array, to know which pixels are real)
        valid_mask:  (1, H, W) bool tensor, True where target is usable
        tile_id:     str, for debugging/logging

    Args:
        root: GAMUS root dir containing images/ and heights/ subfolders.
        split: one of "train", "val", "test".
        manifest_path: optional path to a JSON manifest from
            build_manifest.py, listing tile ids per split. Strongly
            recommended for reproducibility. If omitted, falls back to a
            sorted directory scan (deterministic, but not guaranteed
            stable across different local copies/mirrors of GAMUS).
        crop_size: output (H, W) after resize+crop.
        augment: train-time random crop + flip. Defaults to True for
            split="train", False otherwise.
        target_transform: optional callable applied to the target AFTER
            invalid pixels are zeroed, e.g. torch.log1p. Defaults to
            identity — deliberately not applied automatically, since the
            long-tailed height distribution is a modeling choice.
    """

    def __init__(
        self,
        root: str,
        split: str,
        manifest_path: Optional[str] = None,
        crop_size: Sequence[int] = (518, 518),
        augment: Optional[bool] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        assert split in ("train", "val", "test"), split
        self.root = Path(root)
        self.split = split
        self.crop_size = tuple(crop_size)
        self.augment = augment if augment is not None else (split == "train")
        self.target_transform = target_transform

        self.tile_ids: List[str] = self._load_tile_ids(manifest_path)
        if len(self.tile_ids) == 0:
            raise RuntimeError(
                f"No tiles found for split={split!r} under {self.root}. "
                f"Check `root` and `manifest_path`."
            )
        logger.info("GamusDataset[%s]: %d tiles", split, len(self.tile_ids))

    # ---------- split handling ----------

    def _load_tile_ids(self, manifest_path: Optional[str]) -> List[str]:
        if manifest_path is not None:
            with open(manifest_path) as f:
                manifest = json.load(f)
            return sorted(manifest[self.split])

        img_dir = self.root / "images" / self.split
        rgb_files = sorted(img_dir.glob("*_RGB.h5"))
        return [p.name[: -len("_RGB.h5")] for p in rgb_files]

    # ---------- core dataset API ----------

    def __len__(self):
        return len(self.tile_ids)

    def _paths(self, tile_id: str):
        rgb_path = self.root / "images" / self.split / f"{tile_id}_RGB.h5"
        agl_path = self.root / "heights" / self.split / f"{tile_id}_AGL.h5"
        return rgb_path, agl_path

    def __getitem__(self, idx: int):
        tile_id = self.tile_ids[idx]
        rgb_path, agl_path = self._paths(tile_id)

        rgb = _load_h5_array(rgb_path)     # expected (H, W, 3) uint8
        height = _load_h5_array(agl_path)  # expected (H, W) float32

        rgb = self._validate_rgb(rgb, rgb_path)
        height = self._validate_height(height, agl_path)

        valid_mask = self._build_validity_mask(height)
        height = np.where(valid_mask, height, 0.0).astype(np.float32)

        image_t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
        target_t = torch.from_numpy(height.copy()).unsqueeze(0).float()
        mask_t = torch.from_numpy(valid_mask.copy()).unsqueeze(0).bool()

        image_t, target_t, mask_t = joint_geometric_transform(
            image_t, target_t, mask_t,
            crop_size=self.crop_size,
            train=self.augment,
        )

        image_t = normalize_image(image_t)

        if self.target_transform is not None:
            target_t = self.target_transform(target_t)

        return {
            "image": image_t,
            "target": target_t,
            "valid_mask": mask_t,
            "tile_id": tile_id,
        }

    # ---------- validation helpers ----------

    @staticmethod
    def _validate_rgb(rgb: np.ndarray, path: Path) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(
                f"{path}: expected RGB array shaped (H, W, 3), got {rgb.shape}. "
                f"GAMUS's real HDF5 layout differs from what this module "
                f"assumes — inspect the file and adjust `_validate_rgb`."
            )
        if rgb.dtype != np.uint8:
            logger.warning(
                "%s: RGB dtype is %s, not uint8 as assumed. Rescaling into "
                "[0, 255] uint8 best-effort — verify correctness on real data.",
                path, rgb.dtype,
            )
            if rgb.max() <= 255:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            else:
                rgb = ((rgb.astype(np.float32) / rgb.max()) * 255).astype(np.uint8)
        return rgb

    @staticmethod
    def _validate_height(height: np.ndarray, path: Path) -> np.ndarray:
        if height.ndim == 3 and height.shape[-1] == 1:
            height = height[..., 0]
        if height.ndim != 2:
            raise ValueError(
                f"{path}: expected height array shaped (H, W), got {height.shape}."
            )
        return height.astype(np.float32)

    @staticmethod
    def _build_validity_mask(height: np.ndarray) -> np.ndarray:
        mask = np.isfinite(height)
        for sentinel in NODATA_SENTINELS:
            mask &= ~np.isclose(height, sentinel, atol=1e-3)
        mask &= height >= MIN_VALID_HEIGHT
        return mask


def build_dataloader(
    root: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    manifest_path: Optional[str] = None,
    crop_size: Sequence[int] = (518, 518),
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """Convenience wrapper: GamusDataset -> DataLoader with sane defaults."""
    dataset = GamusDataset(
        root=root, split=split, manifest_path=manifest_path, crop_size=crop_size
    )
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
