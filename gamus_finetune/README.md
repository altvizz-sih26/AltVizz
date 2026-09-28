# gamus_finetune

Model wrapper, loss functions, training loop, and an overfit sanity test
for fine-tuning Depth Anything V2 on GAMUS height (nDSM) data. Depends on
`gamus_dataset/` (sibling folder, unmodified) for data loading, and on
your own local Depth Anything V2 checkout for the pretrained model class.

**Does not modify the DA-V2 repository or its architecture** — `model.py`
imports `DepthAnythingV2` from your checkout and holds it as a submodule.

## Status: verified against real DA-V2 source

`model.py` has been checked directly against the real `depth_anything_v2/
dpt.py` (DPTHead + DepthAnythingV2 classes) and `run.py`, not just
inferred from the general DPT/MiDaS architecture family. Import path,
`ENCODER_CONFIGS` (now including `vitg`), and attribute names
(`model.pretrained`, `model.depth_head`) are all confirmed correct.

Two real issues were found and fixed once the real source was available:
- **Double ReLU floor**: `DPTHead.output_conv2` ends in `Conv2d -> ReLU ->
  Identity`, and `DepthAnythingV2.forward()` applies `F.relu()` again on
  top of that — output was floored at exactly 0, unable to represent the
  small legitimate negative height noise (~-5m) seen in real GAMUS
  samples. `_rebuild_output_head` drops the trailing ReLU/Identity, and
  `GamusDepthModel.forward()` bypasses `DepthAnythingV2.forward()`
  entirely (replicating its two real steps) to skip the second ReLU.
- **Shape mismatch avoided**: `DepthAnythingV2.forward()` returns
  `depth.squeeze(1)` (`(B, H, W)`); bypassing it means `GamusDepthModel`
  returns `depth_head`'s natural `(B, 1, H, W)`, matching GamusDataset's
  target/mask shape with no extra reshaping needed.

`_find_backbone_blocks` remains a search (not a hardcoded path) purely
because it doesn't need to be exact — it just needs to find
`pretrained.blocks`, standard ViT structure — kept generic so a future
encoder variant wouldn't fail silently if it were ever named differently.

`train.py` and `overfit_test.py` both assert `crop_size % 14 == 0` at
startup, since DA-V2's patch embedding requires it and a non-multiple
would otherwise produce a model output smaller than the crop, silently
misaligned with the target/mask, rather than failing loudly.

## Files

- `model.py` — `GamusDepthModel`: wraps pretrained DA-V2, replaces the
  final output conv with a fresh 1-channel layer (plain linear output —
  not forced non-negative, since real GAMUS data has small legitimate
  negative height noise near ground level), plus `freeze_backbone()`,
  `unfreeze_last_n_blocks()`, `get_param_groups()`.
- `losses.py` — `masked_l1`, `gradient_loss` (multi-scale, edge-aware,
  erodes the mask so gradients never mix valid/invalid pixels), and
  `CombinedHeightLoss` combining both with configurable weights.
- `train.py` — full training loop: two-phase freeze schedule
  (`--freeze-epochs`, `--unfreeze-blocks`), AdamW with separate head/
  backbone learning rates, warmup+cosine schedule, mixed precision,
  gradient clipping, checkpointing (`latest.pt` / `best.pt`), and
  validation metrics broken down by height bucket (0–2m / 2–10m / >10m)
  since GAMUS heights are long-tailed and an aggregate metric can hide
  poor performance specifically on tall structures.
- `overfit_test.py` — **not a real training run.** Builds a temp
  directory of symlinks so it can use the real, unmodified
  `GamusDataset` on just 2 (or a few) real tiles, then runs a tight loop
  to confirm loss actually drops to near-zero — i.e. that gradients flow
  and nothing in the model/loss/data wiring is silently broken. Passing
  this says nothing about real model quality; it only says the pipeline
  works.

## Suggested order of use

```bash
# 1. Wiring check — seconds, no real data needed beyond the 2 sample tiles
python overfit_test.py \
    --tiles-dir /path/to/sample_tiles \
    --tile-ids DC_03_26 DC_05_28 \
    --da2-checkpoint /path/to/depth_anything_v2_vits.pth \
    --encoder vits --steps 300

# 2. Real training, once the above passes and you have a real train/val split
python train.py \
    --gamus-root /path/to/GAMUS \
    --manifest /path/to/gamus_manifest.json \
    --da2-checkpoint /path/to/depth_anything_v2_vits.pth \
    --encoder vits \
    --out-dir ./runs/gamus_vits_run1
```

## Not done here (on purpose)

- No fix to `MIN_VALID_HEIGHT` in `gamus_dataset/dataset.py` — flagged
  separately as a small, non-blocking accuracy issue, not yet applied.
- No hyperparameter tuning — `--head-lr`, `--backbone-lr`,
  `--freeze-epochs`, `--l1-weight`/`--grad-weight` are reasonable
  starting defaults, not tuned values. Watch the per-bucket validation
  metrics to decide whether to adjust them.
