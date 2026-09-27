"""
Wraps a pretrained Depth Anything V2 model for height-from-nadir-imagery
fine-tuning, without modifying the DA-V2 repository itself.

Verified against the real `depth_anything_v2/dpt.py` (DPTHead,
DepthAnythingV2 classes) — not a guess anymore. Two things needed fixing
once the real source was available:

1. `DPTHead.output_conv2` ends in `Conv2d(..., 1, ...) -> ReLU -> Identity`,
   and `DepthAnythingV2.forward()` applies `F.relu(depth)` again on top of
   that — the model's output is floored at exactly 0, twice over. Real
   GAMUS height data has small legitimate negative values near ground
   level (down to ~-5m in samples inspected), which a ReLU-floored output
   can never represent. `_rebuild_output_head` replaces the final Conv2d
   and drops the ReLU/Identity after it; `GamusDepthModel.forward` also
   bypasses `DepthAnythingV2.forward()` entirely (replicating its
   `pretrained.get_intermediate_layers` -> `depth_head` steps) to skip the
   second, outer ReLU.
2. `DepthAnythingV2.forward()` returns `depth.squeeze(1)`, i.e. `(B, H, W)`
   — but by calling `depth_head` directly and skipping that squeeze, we
   get `depth_head`'s natural `(B, 1, H, W)` output, which matches
   GamusDataset's target/mask shape without any extra reshaping.

`_find_backbone_blocks` remains a search (not a hardcoded path) since it
only needs to find `pretrained.blocks`, which is standard ViT structure
and unlikely to vary — kept generic mainly so a differently-named backbone
wouldn't fail silently.
"""
import sys

sys.path.append("/Users/tanishka/Desktop/AltVizz/Depth_Wizard/Depth-Anything-V2")
import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# DA-V2 encoder configs (from the real repo's run.py model_configs dict).
ENCODER_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def _find_backbone_blocks(module: nn.Module) -> Optional[nn.ModuleList]:
    """Search for the DINOv2 backbone's transformer block list (`pretrained.blocks`)."""
    for name, child in module.named_modules():
        if name.endswith("blocks") and isinstance(child, nn.ModuleList):
            logger.info("Found backbone block list at '%s' (%d blocks)", name, len(child))
            return child
    return None


def _rebuild_output_head(model: nn.Module) -> nn.Conv2d:
    """
    Rebuilds `depth_head.scratch.output_conv2`, dropping the trailing
    ReLU + Identity that floor DA-V2's original output at zero.

    Real structure (from dpt.py):
        Sequential(
            Conv2d(head_features_1 // 2, head_features_2, 3, 1, 1),
            ReLU(True),
            Conv2d(head_features_2, 1, 1, 1, 0),   # <- replaced
            ReLU(True),                             # <- dropped
            Identity(),                              # <- dropped
        )

    The first Conv+ReLU are kept (legitimate feature processing); only the
    final projection is replaced, with nothing after it, so output can be
    negative.
    """
    try:
        output_conv2 = model.depth_head.scratch.output_conv2
        first_conv = output_conv2[0]
        first_relu = output_conv2[1]
        old_final_conv = output_conv2[2]
    except (AttributeError, IndexError) as e:
        raise RuntimeError(
            "Could not find the expected depth_head.scratch.output_conv2 "
            "structure — DA-V2's architecture differs from dpt.py as "
            "provided. Inspect `model.depth_head` directly and adjust "
            "`_rebuild_output_head`."
        ) from e

    new_final_conv = nn.Conv2d(
        in_channels=old_final_conv.in_channels,
        out_channels=1,
        kernel_size=old_final_conv.kernel_size,
        stride=old_final_conv.stride,
        padding=old_final_conv.padding,
        bias=True,
    )

    model.depth_head.scratch.output_conv2 = nn.Sequential(first_conv, first_relu, new_final_conv)
    logger.info(
        "Rebuilt depth_head.scratch.output_conv2 (in=%d): dropped trailing "
        "ReLU/Identity so output can be negative.",
        old_final_conv.in_channels,
    )
    return new_final_conv


class GamusDepthModel(nn.Module):
    """
    Wraps a pretrained DA-V2 model, holding it as a submodule (never
    subclassing or editing the original classes), and adapts it for
    unbounded height regression in meters.

    `forward()` deliberately does NOT call the wrapped model's own
    `forward()` — it replicates its two real steps
    (`pretrained.get_intermediate_layers` -> `depth_head`) directly, to
    skip the outer `F.relu()` and the final `.squeeze(1)` that
    `DepthAnythingV2.forward()` applies (see module docstring for why).
    """

    def __init__(self, da_v2_model: nn.Module):
        super().__init__()
        self.backbone_and_head = da_v2_model  # unmodified DA-V2 model, held as-is
        self._new_head_layer = _rebuild_output_head(self.backbone_and_head)
        self._backbone_blocks = _find_backbone_blocks(self.backbone_and_head)
        if self._backbone_blocks is None:
            logger.warning(
                "Could not locate a backbone 'blocks' ModuleList — "
                "unfreeze_last_n_blocks() will raise if called. Freezing "
                "the whole backbone still works via freeze_backbone()."
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: (B, 3, H, W), ImageNet-normalized, H and W multiples of 14
        -> (B, 1, H, W) height in meters (can be negative).
        """
        da_v2 = self.backbone_and_head
        patch_h, patch_w = image.shape[-2] // 14, image.shape[-1] // 14

        features = da_v2.pretrained.get_intermediate_layers(
            image, da_v2.intermediate_layer_idx[da_v2.encoder], return_class_token=True
        )
        depth = da_v2.depth_head(features, patch_h, patch_w)  # (B, 1, H, W), no ReLU now
        return depth

    # ---------- freeze/unfreeze schedule ----------

    def freeze_backbone(self) -> None:
        """Freeze every parameter except the new output head layer."""
        head_params = set(id(p) for p in self._new_head_layer.parameters())
        n_frozen = 0
        for p in self.backbone_and_head.parameters():
            if id(p) not in head_params:
                p.requires_grad = False
                n_frozen += 1
        logger.info("Froze %d backbone parameters; head layer left trainable.", n_frozen)

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Unfreeze the last `n` transformer blocks of the backbone."""
        if self._backbone_blocks is None:
            raise RuntimeError(
                "No backbone block list was found at init time — inspect "
                "the real model structure and fix `_find_backbone_blocks`."
            )
        if n <= 0:
            return
        blocks_to_unfreeze = list(self._backbone_blocks)[-n:]
        n_params = 0
        for block in blocks_to_unfreeze:
            for p in block.parameters():
                p.requires_grad = True
                n_params += 1
        logger.info("Unfroze last %d backbone blocks (%d parameters).", n, n_params)

    # ---------- optimizer param groups ----------

    def get_param_groups(self, head_lr: float, backbone_lr: float) -> List[dict]:
        """
        Returns optimizer param groups: the new head layer at `head_lr`,
        and any currently-trainable backbone parameters (e.g. after
        `unfreeze_last_n_blocks`) at `backbone_lr`. Frozen parameters
        (requires_grad=False) are excluded automatically — call this AFTER
        your freeze/unfreeze calls for the current training phase.
        """
        head_param_ids = set(id(p) for p in self._new_head_layer.parameters())
        head_params, backbone_params = [], []
        for p in self.backbone_and_head.parameters():
            if not p.requires_grad:
                continue
            (head_params if id(p) in head_param_ids else backbone_params).append(p)

        groups = [{"params": head_params, "lr": head_lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        return groups


def load_da_v2_model(encoder: str, checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """Loads a pretrained DA-V2 model exactly as run.py does."""
    import sys
    sys.path.append("/Users/tanishka/Desktop/AltVizz/Depth_Wizard/Depth-Anything-V2")
    from depth_anything_v2_from_person1.dpt import DepthAnythingV2  # noqa: import from DA-V2 repo, unmodified

    if encoder not in ENCODER_CONFIGS:
        raise ValueError(f"Unknown encoder {encoder!r}, expected one of {list(ENCODER_CONFIGS)}")

    model = DepthAnythingV2(**ENCODER_CONFIGS[encoder])
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device)


def build_gamus_model(encoder: str, checkpoint_path: str, device: str = "cpu") -> GamusDepthModel:
    """Convenience: load pretrained DA-V2 + wrap it for height regression."""
    da_v2_model = load_da_v2_model(encoder, checkpoint_path, device)
    model = GamusDepthModel(da_v2_model).to(device)
    return model
