"""
Training loop for GAMUS height fine-tuning of Depth Anything V2.

Deliberately does NOT touch the DA-V2 repo or gamus_dataset/ — it only
imports GamusDataset/build_dataloader (unmodified) and the model/loss
wrappers defined alongside this file.

This assumes gamus_dataset/ and gamus_finetune/ sit as sibling folders on
your PYTHONPATH (e.g. both under the same parent directory you run this
from). Adjust the sys.path line below if your layout differs.

Usage:
    python train.py \\
        --gamus-root /path/to/GAMUS \\
        --manifest /path/to/gamus_manifest.json \\
        --da2-checkpoint /path/to/depth_anything_v2_vits.pth \\
        --encoder vits \\
        --out-dir ./runs/gamus_vits_run1
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for gamus_dataset

from gamus_dataset import build_dataloader  # noqa: E402
from gamus_finetune.losses import CombinedHeightLoss  # noqa: E402
from gamus_finetune.model import build_gamus_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------- config ----------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamus-root", required=True)
    ap.add_argument("--manifest", default=None, help="from build_manifest.py; recommended")
    ap.add_argument("--da2-checkpoint", required=True)
    ap.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--crop-size", type=int, default=518)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)

    ap.add_argument("--freeze-epochs", type=int, default=5,
                     help="epochs of head-only training before unfreezing backbone blocks")
    ap.add_argument("--unfreeze-blocks", type=int, default=2,
                     help="number of trailing backbone blocks to unfreeze after freeze-epochs")
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--l1-weight", type=float, default=1.0)
    ap.add_argument("--grad-weight", type=float, default=0.5)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    return ap.parse_args()


# ---------- metrics ----------

HEIGHT_BUCKETS = [(0, 2), (2, 10), (10, float("inf"))]


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> dict:
    """MAE/RMSE overall and per height bucket, computed only over valid pixels."""
    metrics = {}
    diff = (pred - target)

    def mae_rmse(mask):
        n = mask.sum()
        if n == 0:
            return float("nan"), float("nan")
        d = diff[mask]
        return d.abs().mean().item(), d.pow(2).mean().sqrt().item()

    metrics["mae"], metrics["rmse"] = mae_rmse(valid_mask)

    for lo, hi in HEIGHT_BUCKETS:
        bucket_mask = valid_mask & (target >= lo) & (target < hi)
        label = f"{lo}-{hi if hi != float('inf') else 'inf'}m"
        mae, rmse = mae_rmse(bucket_mask)
        metrics[f"mae_{label}"] = mae
        metrics[f"rmse_{label}"] = rmse
        metrics[f"n_{label}"] = int(bucket_mask.sum().item())

    return metrics


# ---------- train / eval loops ----------

def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None, grad_clip=1.0):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, n_batches = 0.0, 0
    agg_metrics = {}

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train_mode):
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                 enabled=(scaler is not None)):
                pred = model(image)
                loss, loss_parts = loss_fn(pred, target, valid_mask)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"]], grad_clip
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"]], grad_clip
                    )
                    optimizer.step()

        with torch.no_grad():
            batch_metrics = compute_metrics(pred.detach(), target, valid_mask)
        for k, v in batch_metrics.items():
            if k.startswith("n_"):
                continue
            agg_metrics.setdefault(k, []).append(v)

        total_loss += loss.item()
        n_batches += 1

    avg_metrics = {k: sum(v) / len(v) for k, v in agg_metrics.items() if v}
    avg_metrics["loss"] = total_loss / max(n_batches, 1)
    return avg_metrics


def make_warmup_cosine(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())
    return LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    assert args.crop_size % 14 == 0, (
        f"--crop-size must be a multiple of 14 (DINOv2 patch size), got {args.crop_size}"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    device = args.device
    logger.info("Using device: %s", device)

    train_loader = build_dataloader(
        root=args.gamus_root, split="train", batch_size=args.batch_size,
        num_workers=args.num_workers, manifest_path=args.manifest,
        crop_size=(args.crop_size, args.crop_size),
    )
    val_loader = build_dataloader(
        root=args.gamus_root, split="val", batch_size=args.batch_size,
        num_workers=args.num_workers, manifest_path=args.manifest,
        crop_size=(args.crop_size, args.crop_size), shuffle=False,
    )

    model = build_gamus_model(args.encoder, args.da2_checkpoint, device=device)
    model.freeze_backbone()

    loss_fn = CombinedHeightLoss(l1_weight=args.l1_weight, grad_weight=args.grad_weight)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device == "cuda") else None

    optimizer = AdamW(model.get_param_groups(args.head_lr, args.backbone_lr))
    total_steps = args.epochs * len(train_loader)
    scheduler = make_warmup_cosine(optimizer, args.warmup_steps, total_steps)

    best_val_mae = float("inf")
    unfroze_yet = False

    for epoch in range(args.epochs):
        if not unfroze_yet and epoch >= args.freeze_epochs:
            model.unfreeze_last_n_blocks(args.unfreeze_blocks)
            optimizer = AdamW(model.get_param_groups(args.head_lr, args.backbone_lr))
            scheduler = make_warmup_cosine(
                optimizer, warmup_steps=100, total_steps=(args.epochs - epoch) * len(train_loader)
            )
            unfroze_yet = True
            logger.info("Epoch %d: unfroze last %d backbone blocks, rebuilt optimizer.",
                        epoch, args.unfreeze_blocks)

        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer, scaler, args.grad_clip)
        for _ in range(len(train_loader)):
            scheduler.step()
        val_metrics = run_epoch(model, val_loader, loss_fn, device)
        dt = time.time() - t0

        logger.info(
            "Epoch %d/%d (%.1fs) — train loss %.4f mae %.3f | val loss %.4f mae %.3f rmse %.3f",
            epoch + 1, args.epochs, dt,
            train_metrics["loss"], train_metrics.get("mae", float("nan")),
            val_metrics["loss"], val_metrics.get("mae", float("nan")),
            val_metrics.get("rmse", float("nan")),
        )
        logger.info("  val per-bucket: %s", {
            k: round(v, 3) for k, v in val_metrics.items() if k.startswith(("mae_", "rmse_"))
        })

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "latest.pt")
        if val_metrics.get("mae", float("inf")) < best_val_mae:
            best_val_mae = val_metrics["mae"]
            torch.save(ckpt, out_dir / "best.pt")
            logger.info("  new best val MAE: %.4f — saved best.pt", best_val_mae)


if __name__ == "__main__":
    main()
