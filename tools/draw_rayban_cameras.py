#!/usr/bin/env python3
"""Mark the Ray-Ban Meta camera modules on every Ray-Ban photo, into ONE folder.

Uses the LATEST detector — YOLOX-Nano (runs/rayban_yolox/best_ckpt.pth, the model
that wins MODEL_COMPARISON.md: recall 0.938 / precision 0.987 at conf 0.40) — to
get the whole-glasses box coordinates, then draws the two camera modules at the
top-outer end-pieces (where the Ray-Ban Meta cameras physically sit), using the
corner geometry from mark_cameras.py.

Unlike mark_cameras.py this collects EVERY Ray-Ban source into a single output
folder instead of per-folder subdirectories.

    PYTHONPATH=. python tools/draw_rayban_cameras.py
    PYTHONPATH=. python tools/draw_rayban_cameras.py --conf 0.25

Sources -> ray_ban_face/, ray_ban_frame/, actual_rayban_cases/
Output  -> camera_marked_rayban/<name>.png
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
# Vendored YOLOX (third_party/YOLOX) — needed to rebuild the YOLOX-Nano model.
sys.path.insert(0, os.path.join(_REPO, "third_party", "YOLOX"))
os.environ.setdefault("YOLOX_EXP", os.path.join(_REPO, "exps", "rayban_yolox_nano.py"))

from infer import build_detector, pick_device, _iter_images  # noqa: E402
from mark_cameras import mark  # noqa: E402

# Latest / best detector per MODEL_COMPARISON.md.
DEFAULT_WEIGHTS = os.path.join(_REPO, "runs", "rayban_yolox", "best_ckpt.pth")
SOURCES = ["ray_ban_face", "ray_ban_frame", "actual_rayban_cases"]
OUT_DIR = os.path.join(_REPO, "camera_marked_rayban")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.40,
                    help="min box confidence (YOLOX-Nano recommended point: 0.40)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--topk", type=int, default=1,
                    help="mark only the K highest-confidence boxes per image")
    ap.add_argument("--all", action="store_true",
                    help="also copy through images where nothing was detected")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        raise SystemExit(f"weights not found: {args.weights}")
    det = build_detector(args.weights, args.device or pick_device())
    print(f"detector={type(det).__name__}  weights={os.path.relpath(args.weights, _REPO)}"
          f"  conf={args.conf}")

    os.makedirs(OUT_DIR, exist_ok=True)
    seen = set()
    n_marked = n_total = 0
    for src in SOURCES:
        abs_src = os.path.join(_REPO, src)
        if not os.path.isdir(abs_src):
            print(f"skip (missing): {src}")
            continue
        for path in _iter_images(abs_src):
            n_total += 1
            boxes = det.detect(path, args.conf)     # -> [(x1,y1,x2,y2,score), ...]
            boxes.sort(key=lambda b: -b[4])
            if args.topk > 0:
                boxes = boxes[:args.topk]
            if not boxes and not args.all:
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            if boxes:
                mark(img, boxes)                    # frame box + camera end-pieces
                n_marked += 1
            stem = os.path.splitext(os.path.basename(path))[0]
            name, k = stem, 1
            while name in seen:
                name, k = f"{stem}_{k}", k + 1
            seen.add(name)
            cv2.imwrite(os.path.join(OUT_DIR, f"{name}.png"), img)
    print(f"marked {n_marked}/{n_total} Ray-Ban photos -> {os.path.relpath(OUT_DIR, _REPO)}/")


if __name__ == "__main__":
    main()
