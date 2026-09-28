"""
Builds a reproducible JSON split manifest for GAMUS by scanning the
directory structure that ships with the dataset (images/{train,val,test}
and heights/{train,val,test}).

NOTE on "official" split: the two published tile counts for GAMUS
disagree (paper: 11,507 tiles / 6304-1059-4144; current Hugging Face
mirror: 5,900 rows / 1200-1600-3100). This script does NOT try to
reconcile that or re-split anything — it just records, deterministically
and with a checksum, exactly which tiles are physically present in each
split directory of *your* downloaded copy, and drops any RGB tile that
has no matching height file (logged as a warning). Use whichever mirror
you actually downloaded, and keep this manifest alongside your results so
the split is reproducible.

Usage:
    python build_manifest.py --root /path/to/GAMUS --out gamus_manifest.json
"""

import argparse
import hashlib
import json
from pathlib import Path


def scan_split(root: Path, split: str):
    img_dir = root / "images" / split
    agl_dir = root / "heights" / split
    if not img_dir.exists():
        raise FileNotFoundError(img_dir)

    rgb_files = sorted(img_dir.glob("*_RGB.h5"))
    tile_ids = []
    missing = []
    for p in rgb_files:
        tile_id = p.name[: -len("_RGB.h5")]
        agl_path = agl_dir / f"{tile_id}_AGL.h5"
        if agl_path.exists():
            tile_ids.append(tile_id)
        else:
            missing.append(tile_id)

    if missing:
        print(
            f"[{split}] WARNING: {len(missing)} RGB tiles have no matching "
            f"height file and were excluded, e.g. {missing[:5]}"
        )
    return tile_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to GAMUS root dir")
    ap.add_argument("--out", default="gamus_manifest.json")
    args = ap.parse_args()

    root = Path(args.root)
    manifest = {}
    for split in ("train", "val", "test"):
        try:
            manifest[split] = scan_split(root, split)
        except FileNotFoundError as e:
            print(f"[{split}] SKIPPED (directory not found): {e}")
            manifest[split] = []
        print(f"[{split}] {len(manifest[split])} tiles")

    payload = json.dumps(manifest, indent=2, sort_keys=True)
    checksum = hashlib.sha256(payload.encode()).hexdigest()[:12]
    Path(args.out).write_text(payload)

    total = sum(len(v) for v in manifest.values())
    print(f"\nWrote {args.out} (checksum {checksum}) — total tiles: {total}")


if __name__ == "__main__":
    main()
