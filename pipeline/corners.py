"""Locate + crop the two TOP-OUTER frame corners at FULL resolution.

The Ray-Ban Meta camera module sits at the top-outer corner / end-piece of the
frame (where the top rim meets the temple hinge) — a small chunky block that an
ordinary glasses corner does not have. The whole-face CNN could not see it (a
160px face crop dissolves the module), so it guessed from pose. This module
crops each top-outer corner from the ORIGINAL full-resolution image, where the
module is actually resolvable, so a CNN can learn module-vs-bare-rim directly.

The right corner is mirrored to the same canonical orientation as the left, so
both sides share training examples and the model is flip-symmetric. A Ray-Ban
carries a module in BOTH corners, so the decision (train_corner_clf) fires only
when both corners read as a module — which a normal-glasses photo (bare rim on
both sides) and the old pose shortcut (a hand near ONE side) cannot satisfy.

Face anchoring reuses pipeline/facedet (Haar). Geometry is face-relative, so it
is scale-invariant; corners are cut from the native-resolution image for detail.
Shared by datasets_corners.py (Python) and mirrored by static/detector/corners.js.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import Config
from pipeline import facedet, region


@dataclass
class Corners:
    left: np.ndarray     # HxWx3 uint8 RGB, canonical orientation — the CENTER box
    right: np.ndarray    # HxWx3 uint8 RGB, MIRRORED to canonical orientation
    face: facedet.Face
    boxes: dict          # {"L": (x,y,w,h), "R": (x,y,w,h)} center box, original px
    grid: dict           # {"L": [rgb...], "R": [rgb...]} all candidate crops
    grid_boxes: dict     # {"L": [(x,y,w,h)...], "R": [...]} candidate boxes, original px


def _one_box(face, side, cy, sz, out, W, H):
    """A single square corner box (original px) for the given size/vertical/outward."""
    w = int(round(sz * face.w))
    top = int(round(cy - w / 2.0))
    if side == "L":
        x = int(round(face.x - out * face.w))
    else:
        x = int(round(face.x + face.w + out * face.w - w))
    x = max(0, min(W - 1, x)); top = max(0, min(H - 1, top))
    w = max(1, min(W - x, w)); h = max(1, min(H - top, w))
    return x, top, w, h


def _candidate_boxes(face, side, eye_cy, cfg, W, H):
    """Grid of candidate top-outer corner boxes for one side (mirror of the
    face-relative grid in pipeline/locate._face_candidates). The module sits at the
    temple hinge just above the eye line, but tilt/framing shift it — so we scan
    sizes x vertical x outward and later take the max P(module). The middle of each
    axis is the canonical single box (index chosen in extract())."""
    boxes = []
    for sz in cfg.corner_grid_sizes:
        for yc in cfg.corner_grid_yc:
            cy = eye_cy - yc * face.h
            for out in cfg.corner_grid_out:
                boxes.append(_one_box(face, side, cy, sz, out, W, H))
    return boxes


def _crop_canonical(image_bgr, box, side, n):
    """Cut a box, mirror the R side to canonical orientation, resize to n, -> RGB."""
    x, y, w, h = box
    crop = image_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return None
    if side == "R":
        crop = cv2.flip(crop, 1)
    interp = cv2.INTER_AREA if (crop.shape[1] > n or crop.shape[0] > n) else cv2.INTER_CUBIC
    crop = cv2.resize(crop, (n, n), interpolation=interp)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _center_box(face, side, eye_cy, cfg, W, H):
    """The canonical single box (middle of each grid axis) — the training positive."""
    sz = cfg.corner_grid_sizes[len(cfg.corner_grid_sizes) // 2]
    yc = cfg.corner_grid_yc[len(cfg.corner_grid_yc) // 2]
    out = cfg.corner_grid_out[len(cfg.corner_grid_out) // 2]
    return _one_box(face, side, eye_cy - yc * face.h, sz, out, W, H)


def extract(image_bgr: np.ndarray, cfg: Config, face: facedet.Face | None = None):
    """Return Corners from a full-resolution BGR image, or None if no face.

    Provides BOTH the canonical center crop (left/right, for the training positive)
    and the full candidate grid per side (grid/grid_boxes, for grid-max inference
    and hard-negative mining). Right side is mirrored to the left's orientation."""
    H, W = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if face is None:
        face = facedet.detect_face(gray)
    if face is None:
        return None
    eye_cy = region._eye_line(gray, face)   # anchor corners to where glasses sit
    n = cfg.corner_input

    boxes, crops, grid, grid_boxes = {}, {}, {}, {}
    for side in ("L", "R"):
        cbox = _center_box(face, side, eye_cy, cfg, W, H)
        center = _crop_canonical(image_bgr, cbox, side, n)
        if center is None:
            return None
        boxes[side] = cbox
        crops[side] = center
        gb = _candidate_boxes(face, side, eye_cy, cfg, W, H)
        gcrops = []
        kept = []
        for b in gb:
            c = _crop_canonical(image_bgr, b, side, n)
            if c is not None:
                gcrops.append(c); kept.append(b)
        grid[side] = gcrops
        grid_boxes[side] = kept

    return Corners(left=crops["L"], right=crops["R"], face=face, boxes=boxes,
                   grid=grid, grid_boxes=grid_boxes)
