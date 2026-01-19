#!/usr/bin/env python3
"""
Download and save flat national flags for all countries.

- Uses pycountry for ISO-3166 country list
- Downloads PNGs from flagcdn.com
- Saves to ./flags_png/xx.png (xx = ISO2 lowercase)
- Skips missing ones (some ISO entries may not exist as flags)
"""

import os
import time
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image
import pycountry


OUT_DIR = Path("flags_png")
SIZE_PX = 256                  # output square size (flags will be padded)
FLAGCDN_WIDTH = 320            # source width (w320)
SLEEP_SEC = 0.05               # be nice to the server
TIMEOUT = 15


def download_flag_png(iso2: str) -> Image.Image | None:
    url = f"https://flagcdn.com/w{FLAGCDN_WIDTH}/{iso2}.png"
    r = requests.get(url, timeout=TIMEOUT)
    if r.status_code != 200:
        return None
    return Image.open(BytesIO(r.content)).convert("RGBA")


def pad_to_square(img: Image.Image, size_px: int) -> Image.Image:
    # keep aspect ratio, pad with transparent background
    w, h = img.size
    scale = size_px / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)

    out = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
    x = (size_px - nw) // 2
    y = (size_px - nh) // 2
    out.paste(img, (x, y))
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok = 0
    missing = 0
    failed = 0

    countries = list(pycountry.countries)

    print(f"Countries in pycountry: {len(countries)}")
    print(f"Saving to: {OUT_DIR.resolve()}")

    for c in countries:
        iso2 = getattr(c, "alpha_2", None)
        if not iso2:
            continue
        iso2 = iso2.lower()

        out_path = OUT_DIR / f"{iso2}.png"
        if out_path.exists():
            ok += 1
            continue

        try:
            img = download_flag_png(iso2)
            if img is None:
                missing += 1
                print(f"[MISS] {iso2}  ({c.name})")
                continue

            img = pad_to_square(img, SIZE_PX)
            img.save(out_path, optimize=True)
            ok += 1
            print(f"[ OK ] {iso2}  ({c.name}) -> {out_path}")

        except Exception as e:
            failed += 1
            print(f"[FAIL] {iso2}  ({c.name}): {e}")

        time.sleep(SLEEP_SEC)

    print("\nDone.")
    print(f"Saved:   {ok}")
    print(f"Missing: {missing}")
    print(f"Failed:  {failed}")


if __name__ == "__main__":
    main()
