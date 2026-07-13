"""Locate the glasses REGION to feed the CNN classifier.

Two localization paths (whichever fits), producing ONE bbox around the whole
glasses so the CNN sees the frame + both end pieces (where a Ray-Ban camera sits):

  * WORN path (face found): a Haar face box + eye/eyeglasses cascades fix the
    eye-line; the region spans the face width across that band.
  * HELD/STUDIO path (no face): the segment() frame bbox (dark frame on a
    lighter background), padded out a little.
  * Fallback: the central 90% crop.

The crop is later resized (with mild aspect distortion) to the CNN input size,
identically at train and inference, so no letterboxing parity to maintain.
Shared by train_region_clf.py (Python) and mirrored by static/detector/region.js.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from config import Config
from pipeline import facedet

_HAAR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", "haar")
_EYE_CACHE: dict = {}


def _eye_cascade(name: str):
    if name not in _EYE_CACHE:
        path = os.path.join(_HAAR_DIR, name)
        if not os.path.exists(path):
            path = cv2.data.haarcascades + name
        _EYE_CACHE[name] = cv2.CascadeClassifier(path)
    return _EYE_CACHE[name]


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int
    method: str          # "worn" | "held" | "center"


def _eye_line(gray: np.ndarray, face: facedet.Face):
    """Median eye-centre y from eye/eyeglasses cascades in the face's upper 62%,
    or a typical eye-line if none fire."""
    x, y, w, h = face.x, face.y, face.w, face.h
    y1 = y + int(0.62 * h)
    roi = gray[max(0, y):y1, max(0, x):x + w]
    if roi.size == 0:
        return y + 0.38 * h
    ms = (max(12, int(0.12 * w)),) * 2
    cys = []
    for name in ("haarcascade_eye_tree_eyeglasses.xml", "haarcascade_eye.xml"):
        for (ex, ey, ew, eh) in _eye_cascade(name).detectMultiScale(roi, 1.1, 4, minSize=ms):
            cys.append(y + ey + eh / 2.0)
    return float(np.median(cys)) if cys else (y + 0.38 * h)


def _clip(x0, y0, x1, y1, W, H):
    x0 = max(0, int(round(x0))); y0 = max(0, int(round(y0)))
    x1 = min(W, int(round(x1))); y1 = min(H, int(round(y1)))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def locate_region(frames, cfg: Config, face=None) -> Region:
    """Return the glasses-region bbox in canonical (preprocessed) pixel coords."""
    H, W = frames.gray.shape[:2]
    if face is None:
        face = facedet.detect_face(frames.gray)

    # ---- WORN: face anchor + eye line ----
    if face is not None:
        eye_cy = _eye_line(frames.gray, face)
        fw, fh = face.w, face.h
        x0 = face.x - cfg.region_wpad * fw
        x1 = face.x + fw + cfg.region_wpad * fw
        y0 = eye_cy - cfg.region_up * fh
        y1 = eye_cy + cfg.region_down * fh
        x, y, w, h = _clip(x0, y0, x1, y1, W, H)
        if w > 20 and h > 20:
            return Region(x, y, w, h, "worn")

    # ---- HELD / STUDIO: dark-frame segmentation ----
    from pipeline import segment as _seg
    seg = _seg.segment(frames.gray_eq, cfg)
    if seg.valid:
        bx, by, bw, bh = seg.bbox
        pad_x = cfg.region_seg_pad * bw
        pad_y = cfg.region_seg_pad * bh
        x, y, w, h = _clip(bx - pad_x, by - pad_y, bx + bw + pad_x, by + bh + pad_y, W, H)
        return Region(x, y, w, h, "held")

    # ---- fallback: central 90% ----
    x, y, w, h = _clip(0.05 * W, 0.05 * H, 0.95 * W, 0.95 * H, W, H)
    return Region(x, y, w, h, "center")


def crop_region(frames, region: Region) -> np.ndarray:
    """BGR crop of the located region from the canonical colour image."""
    return frames.color[region.y:region.y + region.h, region.x:region.x + region.w]
