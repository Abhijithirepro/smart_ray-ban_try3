#!/usr/bin/env python3
"""Detect Ray-Ban Meta glasses and MARK the camera modules, into detected_cameras/.

The detector predicts one box around the whole glasses (class rayban_meta). The two
Ray-Ban Meta camera modules physically sit at the top-outer corners (end pieces) of
that frame, so for every detection we:
  * draw the whole-glasses box (green) with its confidence, and
  * mark each camera location at the top-outer corners with a crosshair + "camera".

Annotated images are written to detected_cameras/ (mirroring the input folder names).

    python mark_cameras.py                          # runs the Ray-Ban folders
    python mark_cameras.py ray_ban_face --conf 0.3  # a specific folder/image
    python mark_cameras.py photo.jpg --weights runs/rayban_yolo/weights/best.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
from infer import build_detector, pick_device, _iter_images, _DEFAULT_YOLO  # noqa: E402

OUT_DIR = os.path.join(_REPO, "detected_cameras")
# default targets: the Ray-Ban folders (where cameras should be found)
DEFAULT_TARGETS = ["ray_ban_face", "ray_ban_frame", "actual_rayban_cases"]

GREEN = (60, 220, 60)
CAM = (40, 40, 235)          # BGR red-ish for camera markers


def camera_points(x1, y1, x2, y2):
    """Top-outer corners of the frame box = the two camera-module endpieces.
    Inset slightly so the marker lands on the endpiece, not the empty corner."""
    w, h = x2 - x1, y2 - y1
    inx, iny = 0.10 * w, 0.13 * h
    return [(int(x1 + inx), int(y1 + iny)), (int(x2 - inx), int(y1 + iny))]


def mark(img, boxes):
    for (x1, y1, x2, y2, score) in boxes:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
        label = f"rayban_meta {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), GREEN, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 40, 10), 2)
        r = max(6, int(0.03 * (x2 - x1)))
        for (cx, cy) in camera_points(x1, y1, x2, y2):
            cv2.circle(img, (cx, cy), r, CAM, 2)
            cv2.drawMarker(img, (cx, cy), CAM, cv2.MARKER_CROSS,
                           markerSize=r, thickness=2)
            cv2.putText(img, "camera", (cx - r, cy - r - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CAM, 2)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", default=None,
                    help="image(s)/folder(s); default: the Ray-Ban folders")
    ap.add_argument("--weights", default=_DEFAULT_YOLO)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--all", action="store_true",
                    help="also save images where nothing was detected (unmarked)")
    ap.add_argument("--topk", type=int, default=1,
                    help="mark only the K highest-confidence boxes per image "
                         "(default 1 — a photo has one pair of glasses; keeps the "
                         "overlay clean when the model emits duplicate boxes)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.weights):
        raise SystemExit(f"weights not found: {args.weights}\n"
                         "train first: python train_detector.py --model yolo")
    det = build_detector(args.weights, args.device or pick_device())
    targets = args.targets or DEFAULT_TARGETS
    os.makedirs(OUT_DIR, exist_ok=True)

    n_marked = n_total = 0
    for t in targets:
        abs_t = t if os.path.isabs(t) else os.path.join(_REPO, t)
        if not os.path.exists(abs_t):
            print(f"skip (missing): {t}", file=sys.stderr)
            continue
        sub = os.path.join(OUT_DIR, os.path.basename(abs_t.rstrip("/"))) \
            if os.path.isdir(abs_t) else OUT_DIR
        os.makedirs(sub, exist_ok=True)
        for path in _iter_images(abs_t):
            n_total += 1
            boxes = det.detect(path, args.conf)
            boxes.sort(key=lambda b: -b[4])
            if args.topk > 0:
                boxes = boxes[:args.topk]
            if not boxes and not args.all:
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            mark(img, boxes)
            cv2.imwrite(os.path.join(sub, os.path.basename(path)), img)
            if boxes:
                n_marked += 1
    print(f"marked {n_marked}/{n_total} images with camera locations -> {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
