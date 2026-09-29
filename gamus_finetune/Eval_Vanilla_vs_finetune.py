"""
Standalone evaluation: vanilla Depth Anything V2 vs GAMUS-fine-tuned DA-V2
on the SAME held-out GAMUS test tiles.

Metrics (computed on valid pixels only): MAE, RMSE, Pearson correlation.

Does NOT modify any training/inference code. It only imports, unmodified:
    gamus_finetune.model.build_gamus_model / load_da_v2_model
    gamus_finetune.dataset.GamusDataset

Four result rows are reported (each individually, then side by side):

    vanilla_raw         vanilla output as-is. It is relative inverse depth
                        (disparity), NOT meters, so MAE/RMSE here are not
                        meaningful. Shown only for completeness.
    vanilla_aligned     vanilla output after a per-image least-squares
                        scale+shift fit to the GT (standard relative-depth
                        protocol). Uses GT to pick the scale, so it is a
                        best case for vanilla.
    finetuned_raw       fine-tuned metric height in meters, no alignment.
                        This is the headline number for the pipeline.
    finetuned_aligned   fine-tuned output with the same per-image fit, for a
                        like-for-like comparison with vanilla_aligned.

Pearson is invariant to scale and shift, so it is identical for the raw and
aligned rows of the same model.

Both models see identical pixels: the dataset's own val/test transform
(resize to input-size, center crop, ImageNet normalize), and the same
validity mask (NaN / nodata / below MIN_VALID_HEIGHT are excluded).

Usage (from anywhere; defaults are your paths):
    python3 eval_vanilla_vs_finetuned.py
    python3 eval_vanilla_vs_finetuned.py --max-tiles 20     # quick dry run

Place this file in gamus_finetune/ (next to infer.py) or in the repo root.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Row definitions
# --------------------------------------------------------------------------
ROWS = [
    ("vanilla_raw", "Vanilla DA-V2, raw output (disparity, NOT meters)"),
    ("vanilla_aligned", "Vanilla DA-V2, per-image scale+shift aligned to GT"),
    ("finetuned_raw", "Fine-tuned DA-V2, raw metric height (m)"),
    ("finetuned_aligned", "Fine-tuned DA-V2, per-image scale+shift aligned to GT"),
]
ROW_LABELS = dict(ROWS)


# --------------------------------------------------------------------------
# Metrics (pure numpy, so they can be tested without torch)
# --------------------------------------------------------------------------
def pearson(p: np.ndarray, g: np.ndarray) -> float:
    """Pearson r between two 1-D arrays; NaN if either is (near) constant."""
    pc = p - p.mean()
    gc = g - g.mean()
    sp = float((pc * pc).sum())
    sg = float((gc * gc).sum())
    if sp < 1e-12 or sg < 1e-12:
        return float("nan")
    return float((pc * gc).sum() / math.sqrt(sp * sg))


def image_metrics(p: np.ndarray, g: np.ndarray):
    """MAE, RMSE, Pearson of p vs g (1-D float64 arrays of valid pixels)."""
    err = p - g
    mae = float(np.abs(err).mean())
    rmse = float(math.sqrt(float((err * err).mean())))
    return mae, rmse, pearson(p, g)


def align_scale_shift(p: np.ndarray, g: np.ndarray):
    """
    Least-squares fit g ~ a*p + b over valid pixels.
    Returns (aligned_pred, a, b). If p is constant, a=0 and b=mean(g).
    The slope is NOT constrained to be positive; a negative slope is
    reported (see the slope diagnostics) instead of being hidden.
    """
    pm, gm = p.mean(), g.mean()
    pc = p - pm
    var = float((pc * pc).mean())
    if var < 1e-12:
        a = 0.0
    else:
        a = float((pc * (g - gm)).mean() / var)
    b = float(gm - a * pm)
    return a * p + b, a, b


class Pooled:
    """Accumulates sums across all valid pixels of all images (float64)."""

    def __init__(self):
        self.n = 0
        self.sp = self.sg = self.spp = self.sgg = self.spg = 0.0
        self.sabs = self.ssq = 0.0

    def add(self, p: np.ndarray, g: np.ndarray):
        err = p - g
        self.n += p.size
        self.sp += float(p.sum())
        self.sg += float(g.sum())
        self.spp += float((p * p).sum())
        self.sgg += float((g * g).sum())
        self.spg += float((p * g).sum())
        self.sabs += float(np.abs(err).sum())
        self.ssq += float((err * err).sum())

    def result(self):
        if self.n == 0:
            return dict(mae=float("nan"), rmse=float("nan"), pearson=float("nan"))
        n = self.n
        mae = self.sabs / n
        rmse = math.sqrt(self.ssq / n)
        vp = self.spp - self.sp * self.sp / n
        vg = self.sgg - self.sg * self.sg / n
        if vp < 1e-12 or vg < 1e-12:
            r = float("nan")
        else:
            r = (self.spg - self.sp * self.sg / n) / math.sqrt(vp * vg)
        return dict(mae=mae, rmse=rmse, pearson=r)


def _nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    return float(x.mean()) if x.size else float("nan")


def _nanmedian(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    return float(np.median(x)) if x.size else float("nan")


def summarize(per_image: list, pooled: Pooled) -> dict:
    """per_image: list of (mae, rmse, pearson) tuples for one row."""
    arr = np.array(per_image, dtype=np.float64) if per_image else np.zeros((0, 3))
    pr = pooled.result()
    return {
        "n_images": int(arr.shape[0]),
        "n_pearson_nan": int(np.isnan(arr[:, 2]).sum()) if arr.size else 0,
        "mae_mean": _nanmean(arr[:, 0]) if arr.size else float("nan"),
        "mae_median": _nanmedian(arr[:, 0]) if arr.size else float("nan"),
        "rmse_mean": _nanmean(arr[:, 1]) if arr.size else float("nan"),
        "rmse_median": _nanmedian(arr[:, 1]) if arr.size else float("nan"),
        "pearson_mean": _nanmean(arr[:, 2]) if arr.size else float("nan"),
        "pearson_median": _nanmedian(arr[:, 2]) if arr.size else float("nan"),
        "pooled_mae": pr["mae"],
        "pooled_rmse": pr["rmse"],
        "pooled_pearson": pr["pearson"],
    }


# --------------------------------------------------------------------------
# Report formatting
# --------------------------------------------------------------------------
def _f(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def build_report(summaries: dict, extras: dict, args) -> str:
    L = []
    L.append("=" * 78)
    L.append("GAMUS held-out evaluation: vanilla DA-V2 vs fine-tuned DA-V2")
    L.append("=" * 78)
    L.append(f"encoder            : {args.encoder}")
    L.append(f"test tiles scored  : {extras['n_scored']} "
             f"(skipped {extras['n_skipped']}: fewer than {args.min_valid_pixels} valid pixels)")
    L.append(f"input size         : {args.input_size}x{args.input_size} (resize + center crop, same for both models)")
    L.append(f"fine-tuned ckpt    : {args.finetuned_checkpoint}")
    L.append(f"vanilla ckpt       : {args.vanilla_checkpoint}")
    L.append("")

    # ---- individual reports ----
    for key, label in ROWS:
        s = summaries[key]
        L.append("-" * 78)
        L.append(label)
        L.append("-" * 78)
        L.append(f"  images                 : {s['n_images']} "
                 f"(Pearson undefined on {s['n_pearson_nan']})")
        L.append(f"  MAE   per-image mean   : {_f(s['mae_mean'])}   median: {_f(s['mae_median'])}   pooled: {_f(s['pooled_mae'])}")
        L.append(f"  RMSE  per-image mean   : {_f(s['rmse_mean'])}   median: {_f(s['rmse_median'])}   pooled: {_f(s['pooled_rmse'])}")
        L.append(f"  Pearson per-image mean : {_f(s['pearson_mean'])}   median: {_f(s['pearson_median'])}   pooled: {_f(s['pooled_pearson'])}")
        if key == "vanilla_raw":
            L.append("  note: output is disparity, so MAE/RMSE above are not in meters and not comparable.")
        if key.endswith("_raw") and key.startswith("finetuned"):
            L.append("  units: meters")
        if key.endswith("_aligned"):
            L.append("  units: meters (after fitting scale+shift to GT per image)")
        L.append("")

    # ---- side by side ----
    L.append("=" * 78)
    L.append("SIDE BY SIDE (per-image mean, median in parentheses)")
    L.append("=" * 78)
    hdr = f"{'model / protocol':<24}{'MAE':>18}{'RMSE':>18}{'Pearson':>18}"
    L.append(hdr)
    short = {
        "vanilla_raw": "vanilla raw*",
        "vanilla_aligned": "vanilla aligned",
        "finetuned_aligned": "fine-tuned aligned",
        "finetuned_raw": "fine-tuned raw (m)",
    }
    for key in ["vanilla_raw", "vanilla_aligned", "finetuned_aligned", "finetuned_raw"]:
        s = summaries[key]
        c1 = f"{_f(s['mae_mean'], 3)} ({_f(s['mae_median'], 3)})"
        c2 = f"{_f(s['rmse_mean'], 3)} ({_f(s['rmse_median'], 3)})"
        c3 = f"{_f(s['pearson_mean'], 3)} ({_f(s['pearson_median'], 3)})"
        L.append(f"{short[key]:<24}{c1:>18}{c2:>18}{c3:>18}")
    L.append("* vanilla raw is disparity, not meters; MAE/RMSE not meaningful.")
    L.append("")
    L.append("SIDE BY SIDE (pooled over all valid pixels)")
    L.append(f"{'model / protocol':<24}{'MAE':>18}{'RMSE':>18}{'Pearson':>18}")
    for key in ["vanilla_raw", "vanilla_aligned", "finetuned_aligned", "finetuned_raw"]:
        s = summaries[key]
        L.append(f"{short[key]:<24}{_f(s['pooled_mae'], 3):>18}{_f(s['pooled_rmse'], 3):>18}{_f(s['pooled_pearson'], 3):>18}")
    L.append("")

    # ---- delta ----
    va, fa = summaries["vanilla_aligned"], summaries["finetuned_aligned"]
    L.append("DELTA, fine-tuned vs vanilla, aligned protocol (per-image mean)")

    def pct(new, old):
        return "nan" if old == 0 or math.isnan(old) or math.isnan(new) else f"{100.0 * (old - new) / old:+.1f}%"

    L.append(f"  MAE     : {_f(va['mae_mean'], 3)} -> {_f(fa['mae_mean'], 3)}   (error reduction {pct(fa['mae_mean'], va['mae_mean'])})")
    L.append(f"  RMSE    : {_f(va['rmse_mean'], 3)} -> {_f(fa['rmse_mean'], 3)}   (error reduction {pct(fa['rmse_mean'], va['rmse_mean'])})")
    dr = fa["pearson_mean"] - va["pearson_mean"]
    L.append(f"  Pearson : {_f(va['pearson_mean'], 3)} -> {_f(fa['pearson_mean'], 3)}   (delta {_f(dr, 3)})")
    L.append("")

    # ---- diagnostics ----
    L.append("DIAGNOSTICS")
    L.append(f"  vanilla tiles with NEGATIVE Pearson (disparity anti-correlated with height): "
             f"{extras['van_neg_r']} / {extras['n_scored']}")
    L.append(f"  vanilla tiles with NEGATIVE fitted slope (alignment flipped the sign)      : "
             f"{extras['van_neg_slope']} / {extras['n_scored']}")
    L.append(f"  fine-tuned fitted slope (should be near 1 if heights are well scaled): "
             f"median {_f(extras['ft_slope_median'], 3)}, mean {_f(extras['ft_slope_mean'], 3)}")
    L.append(f"  fine-tuned fitted shift (should be near 0): median {_f(extras['ft_shift_median'], 3)} m")
    L.append("")
    L.append("Caveats")
    L.append("  - Aligned rows use the GT to fit scale+shift per image, which favors the aligned model.")
    L.append("    The raw fine-tuned row is the only one that measures absolute height error.")
    L.append("  - GT is resized (bilinear) to the input size by the dataset transform; pixels next to")
    L.append("    invalid regions are excluded by the nearest-resized mask (documented approximation).")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Model / data loading (torch imported lazily so metrics are testable alone)
# --------------------------------------------------------------------------
def _add_repo_to_path():
    here = Path(__file__).resolve().parent
    for cand in (here, here.parent):
        if (cand / "gamus_finetune").is_dir():
            sys.path.insert(0, str(cand))
            return cand
    raise RuntimeError(
        "Could not find a 'gamus_finetune' folder next to this script or one level up. "
        "Put this file in gamus_finetune/ or in the repo root (AltVizz/)."
    )


def _pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_torch(path, torch):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def run_eval(args):
    import torch
    from torch.utils.data import DataLoader, Subset

    _add_repo_to_path()
    from gamus_dataset.dataset import GamusDataset # noqa: E402
    from gamus_finetune.model import build_gamus_model, load_da_v2_model  # noqa: E402

    assert args.input_size % 14 == 0, f"--input-size must be a multiple of 14, got {args.input_size}"
    device = _pick_device(args.device)
    print(f"Device: {device}")

    # ---- models ----
    print("Loading vanilla DA-V2 ...")
    vanilla = load_da_v2_model(args.encoder, args.vanilla_checkpoint, device).eval()

    print("Loading fine-tuned model ...")
    finetuned = build_gamus_model(args.encoder, args.vanilla_checkpoint, device)
    ckpt = _load_torch(args.finetuned_checkpoint, torch)
    finetuned.load_state_dict(ckpt["model_state"])
    finetuned.eval()
    print(f"Fine-tuned checkpoint epoch {ckpt.get('epoch', '?')}, "
          f"val_metrics at save time: {ckpt.get('val_metrics', {})}")

    # ---- data ----
    ds = GamusDataset(
        root=args.data_root,
        split="test",
        manifest_path=args.manifest,
        crop_size=(args.input_size, args.input_size),
        augment=False,
    )
    if args.max_tiles is not None:
        ds = Subset(ds, list(range(min(args.max_tiles, len(ds)))))
    print(f"Test tiles: {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    # ---- accumulate ----
    per_image = {k: [] for k, _ in ROWS}
    pooled = {k: Pooled() for k, _ in ROWS}
    csv_rows = []
    van_slopes, ft_slopes, ft_shifts = [], [], []
    n_skipped = 0
    n_done = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            tgt = batch["target"].numpy()          # (B,1,H,W) meters
            msk = batch["valid_mask"].numpy()      # (B,1,H,W) bool
            ids = list(batch["tile_id"])

            ft_out = finetuned(imgs)               # (B,1,H,W) meters, signed
            van_out = vanilla(imgs)                # (B,H,W) relative disparity (ReLU'd)
            if van_out.dim() == 3:
                van_out = van_out.unsqueeze(1)
            assert ft_out.shape == van_out.shape == imgs.new_zeros(tgt.shape).shape, (
                f"shape mismatch: ft {tuple(ft_out.shape)}, vanilla {tuple(van_out.shape)}, "
                f"target {tuple(tgt.shape)}"
            )
            ft_np = ft_out.float().cpu().numpy()
            van_np = van_out.float().cpu().numpy()

            for i, tid in enumerate(ids):
                m = msk[i, 0]
                if int(m.sum()) < args.min_valid_pixels:
                    n_skipped += 1
                    continue
                g = tgt[i, 0][m].astype(np.float64)
                pf = ft_np[i, 0][m].astype(np.float64)
                pv = van_np[i, 0][m].astype(np.float64)
                if not (np.isfinite(pf).all() and np.isfinite(pv).all()):
                    print(f"  WARNING: non-finite prediction in {tid}; skipped")
                    n_skipped += 1
                    continue

                pv_al, a_v, b_v = align_scale_shift(pv, g)
                pf_al, a_f, b_f = align_scale_shift(pf, g)

                preds = {
                    "vanilla_raw": pv,
                    "vanilla_aligned": pv_al,
                    "finetuned_raw": pf,
                    "finetuned_aligned": pf_al,
                }
                row = {
                    "tile_id": tid,
                    "n_valid": int(m.sum()),
                    "gt_mean": float(g.mean()),
                    "gt_max": float(g.max()),
                    "vanilla_slope": a_v,
                    "vanilla_shift": b_v,
                    "finetuned_slope": a_f,
                    "finetuned_shift": b_f,
                }
                for key, p in preds.items():
                    mae, rmse, r = image_metrics(p, g)
                    per_image[key].append((mae, rmse, r))
                    pooled[key].add(p, g)
                    row[f"{key}_mae"] = mae
                    row[f"{key}_rmse"] = rmse
                    row[f"{key}_pearson"] = r
                csv_rows.append(row)
                van_slopes.append(a_v)
                ft_slopes.append(a_f)
                ft_shifts.append(b_f)
                n_done += 1
                if n_done % 25 == 0:
                    print(f"  scored {n_done} tiles ...")

    if n_done == 0:
        raise RuntimeError("No tiles were scored. Check --data-root / --manifest and the validity mask.")

    summaries = {k: summarize(per_image[k], pooled[k]) for k, _ in ROWS}
    van_r = np.array([t[2] for t in per_image["vanilla_raw"]], dtype=np.float64)
    extras = {
        "n_scored": n_done,
        "n_skipped": n_skipped,
        "van_neg_r": int((van_r[~np.isnan(van_r)] < 0).sum()),
        "van_neg_slope": int((np.array(van_slopes) < 0).sum()),
        "ft_slope_median": float(np.median(ft_slopes)),
        "ft_slope_mean": float(np.mean(ft_slopes)),
        "ft_shift_median": float(np.median(ft_shifts)),
    }

    # ---- outputs ----
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = build_report(summaries, extras, args)
    print("\n" + report)
    (outdir / "report.txt").write_text(report)
    (outdir / "summary.json").write_text(json.dumps(
        {"summaries": summaries, "extras": extras,
         "config": {k: str(v) for k, v in vars(args).items()}}, indent=2))
    with open(outdir / "per_image_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    worst = sorted(csv_rows, key=lambda r: -r["finetuned_raw_mae"])[:5]
    print("\nWorst 5 tiles by fine-tuned raw MAE:")
    for r in worst:
        print(f"  {r['tile_id']}: MAE {r['finetuned_raw_mae']:.3f} m, "
              f"GT mean {r['gt_mean']:.2f} m, GT max {r['gt_max']:.1f} m")
    print(f"\nSaved report.txt, summary.json, per_image_metrics.csv to {outdir.resolve()}")


def main():
    ap = argparse.ArgumentParser(description="Vanilla vs fine-tuned DA-V2 on GAMUS test")
    ap.add_argument("--data-root", default="/Users/tanishka/Desktop/AltVizz/gamus_dataset/GAMUS_test")
    ap.add_argument("--manifest", default="/Users/tanishka/Desktop/AltVizz/gamus_test_manifest.json")
    ap.add_argument("--finetuned-checkpoint",
                    default="/Users/tanishka/Desktop/AltVizz/gamus_finetune/runs/gamus_vits_run1/best.pt")
    ap.add_argument("--vanilla-checkpoint",
                    default="/Users/tanishka/Desktop/AltVizz/checkpoints/depth_anything_v2_vits.pth",
                    help="original pretrained DA-V2 .pth; also used to build the fine-tuned architecture")
    ap.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-tiles", type=int, default=None, help="score only the first N test tiles")
    ap.add_argument("--min-valid-pixels", type=int, default=100,
                    help="skip tiles with fewer valid GT pixels than this")
    ap.add_argument("--outdir", default="./eval_results")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    args = ap.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()