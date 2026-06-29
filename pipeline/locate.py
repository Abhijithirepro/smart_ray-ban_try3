"""Stage 3 - locate the two lenses, derive R_lens and the two corner ROIs.

This is the highest-leverage stage: every downstream feature is searched in the
ROI that this module places, at the scale this module derives. ROIs are anchored
to *each detected lens* (not to absolute image position), so the detector is
robust to off-centre framing and left/right flips.

Method A (primary, shape-agnostic): split the frame bbox at its centre; in each
half find the largest interior hole (the lens opening) and take its bbox.
Rimless / filled fallback: use the half-mask centroid with fixed extents.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import Config
from pipeline.segment import Segment


@dataclass
class Lens:
    cx: float
    cy: float
    hw: float   # half-width
    hh: float   # half-height


@dataclass
class CornerROI:
    side: str            # "L" or "R"
    x: int               # ROI top-left in canonical pixels
    y: int
    w: int
    h: int
    cam_cx: float        # expected camera centre (canonical px)
    cam_cy: float


@dataclass
class Located:
    lens_left: Lens
    lens_right: Lens
    r_lens: float
    rois: list           # [CornerROI(L), CornerROI(R)]
    method: str


def _holes_in_region(gray_eq, seg: Segment, cfg: Config):
    """Return (cx, cy, w, h, area) for interior holes of the frame contour."""
    x, y, w, h = seg.bbox
    crop = gray_eq[y:y + h, x:x + w]
    _, th = cv2.threshold(crop, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, hier = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    if hier is None:
        return holes
    hier = hier[0]
    for i, c in enumerate(cnts):
        if hier[i][3] == -1:          # has no parent -> outer contour, skip
            continue
        area = cv2.contourArea(c)
        if area < 0.02 * w * h:        # ignore tiny gaps
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        holes.append((x + bx + bw / 2.0, y + by + bh / 2.0, bw, bh, area))
    return holes


def _lens_from_hole(hole) -> Lens:
    cx, cy, w, h, _ = hole
    return Lens(cx=cx, cy=cy, hw=w / 2.0, hh=h / 2.0)


def _fallback_lens(seg: Segment, side: str) -> Lens:
    """Rimless / filled fallback: half-mask centroid + fixed extents."""
    x, y, w, h = seg.bbox
    half = seg.frame_mask[y:y + h,
                          x:x + w // 2] if side == "L" else \
        seg.frame_mask[y:y + h, x + w // 2:x + w]
    m = cv2.moments(half)
    if m["m00"] > 0:
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
    else:
        cx, cy = (w * 0.25, h * 0.5)
    base_x = x if side == "L" else x + w // 2
    return Lens(cx=base_x + cx, cy=y + cy, hw=0.22 * w, hh=0.50 * h)


def _make_roi(lens: Lens, bbox, side: str, cfg: Config, shape) -> CornerROI:
    """Search region OUTSIDE the lens, on the end piece, where the camera lives.

    It runs from the lens's outer edge (with a small overlap back into the rim)
    outward to the frame edge, and from the top of the frame down to just past
    the lens centre. Anchoring to the lens box makes it scale with the frame
    and follow the actual eye shape, rather than a fixed bbox fraction.
    """
    H, W = shape[:2]
    bx, by, bw, bh = bbox
    overlap = cfg.roi_lens_overlap * lens.hw
    if side == "L":                      # outer edge is the LEFT side
        x0 = bx
        x1 = lens.cx - lens.hw + overlap
    else:                                # outer edge is the RIGHT side
        x0 = lens.cx + lens.hw - overlap
        x1 = bx + bw
    y0 = by
    y1 = lens.cy + cfg.roi_y_below * lens.hh

    # safety: guarantee a minimum width even if lens detection over-reaches
    min_w = cfg.roi_min_w_frac * bw
    if x1 - x0 < min_w:
        if side == "L":
            x1 = x0 + min_w
        else:
            x0 = x1 - min_w

    x0 = max(0, int(round(x0))); y0 = max(0, int(round(y0)))
    x1 = min(W, int(round(x1))); y1 = min(H, int(round(y1)))
    cam_cx = (x0 + x1) / 2.0             # region centre (weak prior only)
    cam_cy = (y0 + y1) / 2.0
    return CornerROI(side=side, x=x0, y=y0, w=x1 - x0, h=y1 - y0,
                     cam_cx=cam_cx, cam_cy=cam_cy)


def locate(gray_eq: np.ndarray, seg: Segment, cfg: Config) -> Located:
    x, y, w, h = seg.bbox
    cx_split = x + w / 2.0
    method = "holes"

    holes = _holes_in_region(gray_eq, seg, cfg)
    left_holes = [hl for hl in holes if hl[0] < cx_split]
    right_holes = [hl for hl in holes if hl[0] >= cx_split]

    if left_holes:
        lens_l = _lens_from_hole(max(left_holes, key=lambda z: z[4]))
    else:
        lens_l = _fallback_lens(seg, "L"); method = "holes+fallback"
    if right_holes:
        lens_r = _lens_from_hole(max(right_holes, key=lambda z: z[4]))
    else:
        lens_r = _fallback_lens(seg, "R"); method = "holes+fallback"

    r_lens = float(np.mean([lens_l.hw, lens_r.hw]))

    rois = [_make_roi(lens_l, seg.bbox, "L", cfg, gray_eq.shape),
            _make_roi(lens_r, seg.bbox, "R", cfg, gray_eq.shape)]
    return Located(lens_left=lens_l, lens_right=lens_r, r_lens=r_lens,
                   rois=rois, method=method)
