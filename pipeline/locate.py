"""Stage 3 - locate the two top-outer corner ROIs the classifier scores.

Two localization paths (whichever produces scorable corners wins):

  * FACE path (worn glasses): a Haar face box (pipeline/facedet) anchors a small
    grid of candidate corner crops per side; features scores them all and keeps
    the peak P(camera). Pure CV can't find glasses on a face, so the face is the
    anchor. Placement imprecision is absorbed by searching the grid.
  * HELD path (glasses held up / studio): the original lens-hole method - split
    the frame bbox at its centre, take the largest interior hole per half as the
    lens, and anchor one corner ROI outside each lens on the end piece.

`two_lens_gate` provides the held-path domain gate (a plausible two-lens frame);
the face path is gated simply by "a face was found".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from config import Config
from pipeline.segment import Segment
from pipeline.facedet import Face


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
    path: str                        # "face" | "held"
    candidates: dict                 # {"L": [CornerROI...], "R": [CornerROI...]}
    method: str
    face: Face | None = None
    lens_left: Lens | None = None
    lens_right: Lens | None = None
    r_lens: float = 0.0
    rois: list = field(default_factory=list)   # best/first ROI per side (viz)


# --------------------------------------------------------------------------- #
# FACE path (worn glasses)
# --------------------------------------------------------------------------- #
def _clip_roi(x, y, w, h, side, shape) -> CornerROI:
    H, W = shape[:2]
    x0 = max(0, int(round(x))); y0 = max(0, int(round(y)))
    x1 = min(W, int(round(x + w))); y1 = min(H, int(round(y + h)))
    return CornerROI(side=side, x=x0, y=y0, w=max(0, x1 - x0), h=max(0, y1 - y0),
                     cam_cx=(x0 + x1) / 2.0, cam_cy=(y0 + y1) / 2.0)


def _face_candidates(face: Face, side: str, cfg: Config, shape) -> list:
    """Grid of candidate corner crops in one top-outer region of the face box."""
    out = []
    for sz in cfg.face_search_sizes:
        w = face.w * sz
        h = w
        for dy in cfg.face_search_dy:
            cy = face.y + face.h * dy
            for dx in cfg.face_search_dx:
                if side == "L":
                    x = face.x + face.w * dx
                else:
                    x = face.x + face.w - w - face.w * dx
                out.append(_clip_roi(x, cy - h / 2.0, w, h, side, shape))
    return out


def locate_face(face: Face, cfg: Config, shape) -> Located:
    cand = {"L": _face_candidates(face, "L", cfg, shape),
            "R": _face_candidates(face, "R", cfg, shape)}
    rois = [cand["L"][0], cand["R"][0]]        # placeholder; features picks best
    return Located(path="face", candidates=cand, method=f"haar:{face.method}",
                   face=face, rois=rois)


# --------------------------------------------------------------------------- #
# HELD path (glasses held up / studio) - original lens-hole method
# --------------------------------------------------------------------------- #
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

    Runs from the lens's outer edge (small overlap back into the rim) outward,
    but capped at `roi_outer_max * lens.hw` beyond the lens edge so it can't
    slide onto a hand/hair when the frame bbox is large.
    """
    H, W = shape[:2]
    bx, by, bw, bh = bbox
    overlap = cfg.roi_lens_overlap * lens.hw
    outer_cap = cfg.roi_outer_max * lens.hw
    if side == "L":                      # outer edge is the LEFT side
        x0 = max(bx, lens.cx - lens.hw - outer_cap)
        x1 = lens.cx - lens.hw + overlap
    else:                                # outer edge is the RIGHT side
        x0 = lens.cx + lens.hw - overlap
        x1 = min(bx + bw, lens.cx + lens.hw + outer_cap)
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


def locate_held(gray_eq: np.ndarray, seg: Segment, cfg: Config) -> Located:
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
    roi_l = _make_roi(lens_l, seg.bbox, "L", cfg, gray_eq.shape)
    roi_r = _make_roi(lens_r, seg.bbox, "R", cfg, gray_eq.shape)
    return Located(path="held", candidates={"L": [roi_l], "R": [roi_r]},
                   method=method, lens_left=lens_l, lens_right=lens_r,
                   r_lens=r_lens, rois=[roi_l, roi_r])


def locate(gray_eq: np.ndarray, seg: Segment, cfg: Config,
           face: Face | None = None) -> Located:
    """Face box present -> worn path; else held-glasses lens path."""
    if face is not None:
        return locate_face(face, cfg, gray_eq.shape)
    return locate_held(gray_eq, seg, cfg)


# --------------------------------------------------------------------------- #
# Held-path domain gate: a plausible, tilt-tolerant two-lens frame
# --------------------------------------------------------------------------- #
def two_lens_gate(loc: Located, cfg: Config):
    """Return (passed: bool, reason: str) - the domain gate.

    We deliberately do NOT gate on lens geometry: the held-path lens boxes are too
    noisy (real Ray-Bans whose BOTH corners fire 1.00 were being vetoed for
    "implausible separation"), and gating on studio assumptions is exactly what
    made the original detector reject real photos. FP control rests on the
    both-corners classifier rule + the (retrained) discriminative classifier, which
    already holds normal_frame at 25/25 NORMAL. The gate only confirms structure is
    present: a face was anchored, or two lens regions were located.
    """
    if loc.path == "face":
        return (loc.face is not None), ("face anchor" if loc.face else "no face")
    if loc.lens_left is None or loc.lens_right is None:
        return False, "no two lenses found"
    return True, "held glasses frame"
