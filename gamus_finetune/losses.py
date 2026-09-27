"""
Loss functions for GAMUS height regression. Every function takes
(pred, target, valid_mask) with matching (B, 1, H, W) shapes — the exact
contract GamusDataset + GamusDepthModel already produce — and computes
its result only over valid pixels.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def _safe_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of x over mask==True; returns 0 (not NaN) if mask is empty."""
    n_valid = mask.sum()
    if n_valid == 0:
        return x.new_tensor(0.0)
    return x[mask].sum() / n_valid


def masked_l1(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over valid pixels only."""
    return _safe_mean((pred - target).abs(), valid_mask)


def _erode_mask_1px(mask: torch.Tensor) -> torch.Tensor:
    """
    Shrinks a boolean validity mask by 1 pixel in each direction, so
    gradient computations (which combine a pixel with its neighbor) never
    mix a valid pixel with an invalid one and call the result "valid".
    """
    m = mask.float()
    eroded = F.max_pool2d(1 - m, kernel_size=3, stride=1, padding=1)
    return (1 - eroded).bool()


def gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    scales: Tuple[int, ...] = (1, 2),
) -> torch.Tensor:
    """
    Multi-scale finite-difference gradient matching loss (as used in
    MiDaS/DA-V2-family training) — encourages sharp height discontinuities
    at object boundaries (e.g. building edges), which plain L1 undervalues
    since edge pixels are a small minority of any tile.

    `scales`: downsampling factors to compute gradients at, e.g. (1, 2)
    means "at full resolution" and "at half resolution" — coarser scales
    catch larger structures, finer scales catch sharp edges.
    """
    mask = _erode_mask_1px(valid_mask)
    total = pred.new_tensor(0.0)

    for scale in scales:
        if scale > 1:
            p = F.avg_pool2d(pred, kernel_size=scale, stride=scale)
            t = F.avg_pool2d(target, kernel_size=scale, stride=scale)
            m = F.max_pool2d(mask.float(), kernel_size=scale, stride=scale) > 0.999
        else:
            p, t, m = pred, target, mask

        pred_dx = p[:, :, :, 1:] - p[:, :, :, :-1]
        pred_dy = p[:, :, 1:, :] - p[:, :, :-1, :]
        target_dx = t[:, :, :, 1:] - t[:, :, :, :-1]
        target_dy = t[:, :, 1:, :] - t[:, :, :-1, :]

        mask_dx = m[:, :, :, 1:] & m[:, :, :, :-1]
        mask_dy = m[:, :, 1:, :] & m[:, :, :-1, :]

        total = total + _safe_mean((pred_dx - target_dx).abs(), mask_dx)
        total = total + _safe_mean((pred_dy - target_dy).abs(), mask_dy)

    return total / len(scales)


class CombinedHeightLoss:
    """
    total = l1_weight * masked_l1 + grad_weight * gradient_loss

    Both weights are constructor args, not hardcoded, so they're easy to
    sweep. Defaults (1.0, 0.5) are a reasonable starting point, not a
    tuned result — GAMUS's long-tailed height distribution (mostly
    ground/low vegetation, rare tall buildings) means the right balance
    is worth checking against your own validation metrics, particularly
    the per-height-bucket breakdown (see train.py), rather than trusting
    these defaults blindly.
    """

    def __init__(self, l1_weight: float = 1.0, grad_weight: float = 0.5,
                 grad_scales: Tuple[int, ...] = (1, 2)):
        self.l1_weight = l1_weight
        self.grad_weight = grad_weight
        self.grad_scales = grad_scales

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor):
        l1 = masked_l1(pred, target, valid_mask)
        grad = gradient_loss(pred, target, valid_mask, scales=self.grad_scales)
        total = self.l1_weight * l1 + self.grad_weight * grad
        return total, {"l1": l1.item(), "grad": grad.item(), "total": total.item()}
