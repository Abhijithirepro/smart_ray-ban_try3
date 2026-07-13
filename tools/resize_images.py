#!/usr/bin/env python3
"""Resize the main source images to exactly 640x480 (letterbox, no distortion).

NON-DESTRUCTIVE: originals are left untouched; resized copies are written to
data/resized_640x480/<source-folder>/<file>. Each image is scaled to fit inside
640x480 preserving its aspect ratio, then padded with black bars to the exact size
(so glasses keep their real shape — no stretching).

    python tools/resize_images.py                       # main photo folders
    python tools/resize_images.py --include-meglass      # + the 500 MeGlass negs
    python tools/resize_images.py --width 640 --height 480 --pad 114 114 114
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from pipeline.preprocess import load_image, _flatten_alpha  # noqa: E402

OUT_ROOT = os.path.join(_REPO, "data", "resized_640x480")
EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
# the images the user curated by hand (positives + in-domain negatives)
MAIN_FOLDERS = ["ray_ban_face", "ray_ban_frame", "actual_rayban_cases",
                "normal_frame", "normal_glassess"]


def letterbox(img, tw, th, pad):
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((th, tw, 3), pad, dtype=np.uint8)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def _files(folder):
    out = []
    for root, _, names in os.walk(folder):
        for n in sorted(names):
            if os.path.splitext(n)[1].lower() in EXTS:
                out.append(os.path.join(root, n))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--pad", type=int, nargs=3, default=[0, 0, 0],
                    metavar=("B", "G", "R"), help="pad colour BGR (default black)")
    ap.add_argument("--include-meglass", action="store_true",
                    help="also resize data/normal/meglass (500 bulk negatives)")
    ap.add_argument("--folders", nargs="*", default=None,
                    help="override which source folders to resize")
    args = ap.parse_args(argv)

    folders = args.folders or list(MAIN_FOLDERS)
    if args.include_meglass:
        folders.append("data/normal/meglass")

    total = errors = 0
    for folder in folders:
        abs_folder = os.path.join(_REPO, folder)
        if not os.path.isdir(abs_folder):
            print(f"skip (missing): {folder}", file=sys.stderr)
            continue
        n = 0
        for path in _files(abs_folder):
            try:
                img = _flatten_alpha(load_image(path))
                out = letterbox(img, args.width, args.height, args.pad)
            except Exception as e:  # noqa: BLE001
                print(f"  error: {path} :: {e}", file=sys.stderr)
                errors += 1
                continue
            rel = os.path.relpath(path, abs_folder)
            dst = os.path.join(OUT_ROOT, folder, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # keep .jpg/.png as-is; write jpg for jpeg-family, png otherwise
            cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n += 1
            total += 1
        print(f"{folder}: {n} images -> data/resized_640x480/{folder}/")

    print(f"\ndone: {total} images at {args.width}x{args.height} "
          f"(letterboxed) in {OUT_ROOT}  (errors: {errors})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
