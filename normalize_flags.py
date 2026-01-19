#!/usr/bin/env python3
"""
normalize_flags.py

Reads downloaded PNG flags (RGBA), crops away transparent padding, stretches
content to canonical 2:3 aspect ratio, and saves:

1) A single numpy array:
    flags_2x3_uint8.npy  -> shape [N, H, W, 3], dtype uint8

2) Metadata:
    flags_2x3_meta.json  -> list of dicts with iso2 + crop info

3) Normalized PNGs for inspection:
    flags_norm_2x3/xx.png

Expected input:
    flags_png/xx.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


# ----------------------------
# Config
# ----------------------------

IN_DIR = Path("flags_png")

OUT_NPY = Path("flags_2x3_uint8.npy")
OUT_META = Path("flags_2x3_meta.json")

OUT_NORM_DIR = Path("flags_norm_2x3")

# Canonical 2:3 (H:W = 2:3)
H = 60
W = 90

ALPHA_THRESH = 5   # consider pixels with alpha > this as "non-transparent"
CROP_MARGIN = 2    # pixels around bbox
RESAMPLE = Image.Resampling.LANCZOS


# ----------------------------
# Helpers
# ----------------------------

def compute_alpha_bbox(alpha: np.ndarray, thresh: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Returns bbox (x0, y0, x1, y1) inclusive-exclusive for alpha > thresh.
    If no pixels pass threshold, returns None.
    """
    ys, xs = np.where(alpha > thresh)
    if len(xs) == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return (x0, y0, x1, y1)


def expand_bbox(bbox: Tuple[int, int, int, int], w: int, h: int, margin: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(w, x1 + margin)
    y1 = min(h, y1 + margin)
    return (x0, y0, x1, y1)


def load_and_normalize_flag(path: Path) -> Tuple[np.ndarray, dict, Image.Image]:
    """
    Loads one PNG flag and returns:
      - rgb_uint8: np.ndarray [H, W, 3]
      - meta: dict
      - norm_img_rgb: PIL.Image RGB (W,H)
    """
    img = Image.open(path).convert("RGBA")
    orig_w, orig_h = img.size

    rgba = np.array(img, dtype=np.uint8)  # [orig_h, orig_w, 4]
    alpha = rgba[..., 3]

    bbox = compute_alpha_bbox(alpha, ALPHA_THRESH)
    if bbox is None:
        raise ValueError("no non-transparent pixels found (empty alpha bbox)")

    bbox2 = expand_bbox(bbox, orig_w, orig_h, CROP_MARGIN)
    x0, y0, x1, y1 = bbox2

    cropped = img.crop((x0, y0, x1, y1))  # RGBA
    crop_w, crop_h = cropped.size

    # Stretch to canonical 2:3
    norm_img_rgb = cropped.resize((W, H), RESAMPLE).convert("RGB")
    rgb = np.array(norm_img_rgb, dtype=np.uint8)  # [H, W, 3]

    iso2 = path.stem.lower()

    meta = {
        "iso2": iso2,
        "filename": path.name,
        "orig_size": [orig_h, orig_w],     # [H, W]
        "crop_bbox": [x0, y0, x1, y1],     # PIL crop box
        "crop_size": [crop_h, crop_w],     # [H, W]
        "out_size": [H, W],                # [H, W]
    }

    return rgb, meta, norm_img_rgb


# ----------------------------
# Main
# ----------------------------

def main():
    if not IN_DIR.exists():
        raise SystemExit(f"Input directory not found: {IN_DIR.resolve()}")

    paths = sorted(IN_DIR.glob("*.png"))
    if not paths:
        raise SystemExit(f"No PNGs found in: {IN_DIR.resolve()}")

    OUT_NORM_DIR.mkdir(parents=True, exist_ok=True)

    flags_rgb = []
    meta_list = []

    skipped = 0

    for p in paths:
        try:
            rgb, meta, norm_img = load_and_normalize_flag(p)

            # Save normalized PNG for inspection
            out_png = OUT_NORM_DIR / f"{meta['iso2']}.png"
            norm_img.save(out_png, optimize=True)

            flags_rgb.append(rgb)
            meta_list.append(meta)

            print(f"[ OK ] {meta['iso2']} -> {rgb.shape}  saved={out_png}")
        except Exception as e:
            skipped += 1
            print(f"[SKIP] {p.name}: {e}")

    if not flags_rgb:
        raise SystemExit("No flags processed successfully.")

    arr = np.stack(flags_rgb, axis=0)  # [N, H, W, 3]
    arr = np.ascontiguousarray(arr, dtype=np.uint8)

    np.save(OUT_NPY, arr)

    with OUT_META.open("w", encoding="utf-8") as f:
        json.dump(meta_list, f, indent=2)

    print("\nDone.")
    print(f"Saved: {OUT_NPY.resolve()}  shape={arr.shape} dtype={arr.dtype}")
    print(f"Saved: {OUT_META.resolve()}  entries={len(meta_list)}")
    print(f"Saved normalized PNGs to: {OUT_NORM_DIR.resolve()}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
