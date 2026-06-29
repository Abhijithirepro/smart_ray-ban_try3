"""Stage 4 - per-corner features (computed identically for the L and R ROI).

A camera module looks like: a small, dark, near-circular glassy element with a
metallic bezel and (usually) a tiny bright specular glint, set in a chunky
corner of the frame. We measure several *independent* cues so no single
failure mode (black-on-black, a stray screw, a reflection) dominates:

  (a) f_circle  - a Hough circle exists at the module location
  (b) f_blob    - a dark, circular blob exists (relative threshold -> contours)
  (c) f_dark    - the module is darker than the surrounding frame ring
  (d) f_spec    - a small bright specular cluster sits inside the module
  (e) circ      - raw circularity (kept for debug)
  (f) f_thick   - the corner of the frame is chunky (distance transform)

Module localisation is the crux: the ROI is broad, so several round things can
appear in it (lens edge, logo, reflection). We gather circle/blob candidates,
then pick the one that best combines (i) a position prior toward the expected
top-outer-corner camera spot, (ii) local darkness, and (iii) a specular glint.
The white background is excluded everywhere via the filled frame mask.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import Config
from pipeline.locate import CornerROI, Located
from pipeline.segment import Segment


@dataclass
class CornerFeatures:
    side: str
    f_circle: float = 0.0
    f_blob: float = 0.0
    f_dark: float = 0.0
    f_spec: float = 0.0
    f_thick: float = 0.0
    circ: float = 0.0
    # debug geometry (canonical px)
    circle: tuple | None = None          # chosen (cx, cy, r) of the module
    hough_candidates: list = field(default_factory=list)  # all (cx,cy,r)
    spec_pixels: list = field(default_factory=list)        # [(x,y), ...]
    r_window: tuple = (0.0, 0.0)


@dataclass
class _Cand:
    cx: float       # ROI-local
    cy: float
    r: float
    source: str     # "hough" | "blob"
    circ: float = 0.0


def _radius_window(frame_w: float, cfg: Config) -> tuple:
    r_min = max(cfg.cam_r_min_px, cfg.cam_r_min_w * frame_w)
    r_max = cfg.cam_r_max_w * frame_w
    return r_min, r_max


def _hough_circles(roi_blur, r_min, r_max, cfg: Config):
    min_dist = max(5, int(cfg.hc_min_dist_frac * roi_blur.shape[1]))
    circles = cv2.HoughCircles(
        roi_blur, cv2.HOUGH_GRADIENT, dp=cfg.hc_dp, minDist=min_dist,
        param1=cfg.hc_param1, param2=cfg.hc_param2,
        minRadius=int(max(1, r_min)), maxRadius=int(max(2, r_max)))
    if circles is None:
        return []
    return [(float(c[0]), float(c[1]), float(c[2]))
            for c in np.round(circles[0])]


def _dark_blobs(roi_gray, roi_fg, r_min, r_max, cfg: Config):
    """All dark, in-window blobs (on the frame), with their circularity."""
    vals = roi_gray[roi_fg > 0]
    if vals.size < 16:
        return []
    mean, std = float(vals.mean()), float(vals.std())
    thr = mean - cfg.dark_k * std
    dark = ((roi_gray < thr) & (roi_fg > 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < math.pi * (r_min * 0.6) ** 2:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        circ = 4.0 * math.pi * area / (peri * peri)
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r_min <= r <= r_max:
            out.append(_Cand(cx, cy, r, "blob", circ))
    return out


def _glint_candidates(roi_gray, roi_fg, r_min, r_max, cfg: Config):
    """Seed a candidate at every small bright cluster sitting on the frame.

    The camera module's specular glint is its most reliable signature on a
    black frame (where darkness/contour cues collapse). We grow each glint into
    a module-sized disk so darkness/circle validation can run on it.
    """
    fg = roi_fg > 0
    vals = roi_gray[fg]
    if vals.size < 16:
        return []
    fmax = int(vals.max())
    thr = min(cfg.glint_seed_thr, fmax - cfg.glint_seed_below_max)
    bright = ((roi_gray >= thr) & fg).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(bright, connectivity=8)
    r_mid = 0.5 * (r_min + r_max)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (2 <= area <= cfg.glint_seed_max_area):
            continue
        cx, cy = cents[i]
        out.append(_Cand(float(cx), float(cy), float(r_mid), "glint"))
    # also seed at the single brightest on-frame pixel (the camera glass is
    # glossy; its glint is often the brightest point in a matte-frame corner)
    masked = np.where(fg, roi_gray, 0)
    _, _, _, maxloc = cv2.minMaxLoc(masked.astype(np.uint8))
    out.append(_Cand(float(maxloc[0]), float(maxloc[1]), float(r_mid), "glint"))
    return out


def _local_hough(roi_blur, cx, cy, r_min, r_max, cfg: Config):
    """Refine: run a sensitive Hough in a tight patch around (cx, cy).

    Returns the circle nearest the patch centre, in ROI coords, or None.
    """
    pad = int(2.2 * r_max)
    H, W = roi_blur.shape[:2]
    x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
    x1, y1 = min(W, int(cx + pad)), min(H, int(cy + pad))
    patch = roi_blur[y0:y1, x0:x1]
    if patch.shape[0] < 6 or patch.shape[1] < 6:
        return None
    circles = cv2.HoughCircles(
        patch, cv2.HOUGH_GRADIENT, dp=cfg.hc_dp, minDist=max(5, int(r_min)),
        param1=cfg.hc_param1, param2=cfg.hc_param2_local,
        minRadius=int(max(1, r_min)), maxRadius=int(max(2, r_max)))
    if circles is None:
        return None
    pcx, pcy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    best = min(circles[0], key=lambda c: (c[0] - pcx) ** 2 + (c[1] - pcy) ** 2)
    return (x0 + float(best[0]), y0 + float(best[1]), float(best[2]))


def _disk_mask(shape, cx, cy, r):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, (int(round(cx)), int(round(cy))), max(1, int(round(r))), 255, -1)
    return m


def _ring_median(roi_gray, roi_fg, cx, cy, r):
    inner = _disk_mask(roi_gray.shape, cx, cy, r * 1.4)
    outer = _disk_mask(roi_gray.shape, cx, cy, r * 2.6)
    ring = cv2.subtract(outer, inner)
    ring = cv2.bitwise_and(ring, roi_fg)   # frame only, not background
    v = roi_gray[ring > 0]
    return float(np.median(v)) if v.size else float(roi_gray.mean())


def _darkness(roi_gray, roi_fg, cx, cy, r, cfg: Config):
    """Is there a dark glassy CORE darker than the surrounding frame ring?

    Uses the 20th-percentile of the disk (the dark glass), not the mean, so a
    bright specular glint inside the module does not mask the darkness.
    """
    disk = _disk_mask(roi_gray.shape, cx, cy, r)
    inner = roi_gray[disk > 0]
    if inner.size == 0:
        return 0.0
    glass = float(np.percentile(inner, 20))
    ring_med = _ring_median(roi_gray, roi_fg, cx, cy, r)
    return float(np.clip((ring_med - glass) / cfg.dark_norm, 0.0, 1.0))


def _glint(roi_gray, roi_fg, cx, cy, r, cfg: Config):
    """Return (has_glint, [pixel coords]) for a CENTRAL specular cluster.

    The camera glass throws a glint near the optical centre; we require the
    bright pixels to sit within glint_central_frac * r of the module centre.
    This rejects bright lens-opening edges that merely clip the disk rim.
    """
    core = _disk_mask(roi_gray.shape, cx, cy, cfg.glint_central_frac * r) > 0
    fg = roi_fg > 0
    frame_vals = roi_gray[fg]
    if frame_vals.size == 0:
        return False, []
    frame_max = int(frame_vals.max())
    bright_thr = max(cfg.spec_abs_min, frame_max - cfg.spec_rel_below_max)
    bright = (roi_gray >= bright_thr) & core & fg
    n = int(bright.sum())
    disk_area = max(1, int((_disk_mask(roi_gray.shape, cx, cy, r) > 0).sum()))
    if cfg.spec_min_px <= n <= cfg.spec_max_area_frac * disk_area:
        ys, xs = np.where(bright)
        return True, list(zip(xs.tolist(), ys.tolist()))
    return False, []


def _corner_features(roi: CornerROI, frames, seg: Segment, loc: Located,
                     cfg: Config) -> CornerFeatures:
    out = CornerFeatures(side=roi.side)
    if roi.w < 6 or roi.h < 6:
        return out

    frame_w = seg.bbox[2]
    roi_gray = frames.gray[roi.y:roi.y + roi.h, roi.x:roi.x + roi.w]
    roi_blur = frames.gray_blur[roi.y:roi.y + roi.h, roi.x:roi.x + roi.w]
    roi_fg = seg.frame_mask[roi.y:roi.y + roi.h, roi.x:roi.x + roi.w]
    r_min, r_max = _radius_window(frame_w, cfg)
    out.r_window = (round(r_min, 1), round(r_max, 1))

    # candidate camera modules ---------------------------------------------
    hough = _hough_circles(roi_blur, r_min, r_max, cfg)
    out.hough_candidates = [(roi.x + c[0], roi.y + c[1], c[2]) for c in hough]
    cands = [_Cand(c[0], c[1], c[2], "hough") for c in hough]
    cands += _dark_blobs(roi_gray, roi_fg, r_min, r_max, cfg)
    cands += _glint_candidates(roi_gray, roi_fg, r_min, r_max, cfg)
    # snap each glint candidate's radius to a coincident Hough circle if any
    for c in cands:
        if c.source != "glint":
            continue
        for (hx, hy, hr) in hough:
            if math.hypot(hx - c.cx, hy - c.cy) <= c.r:
                c.r = hr
                break
    if not cands:
        out.f_thick = _thickness(roi, seg, loc, cfg)
        return out

    # --- signature-first selection ----------------------------------------
    # Score every candidate by HOW CAMERA-LIKE it is (bezel + central glint +
    # dark glassy core), not by where it sits. A weak region-centre prior only
    # breaks ties between equally camera-like candidates. This is what lets the
    # detector generalise: the camera is defined by appearance, not position.
    rx = roi.w / 2.0
    ry = roi.h / 2.0

    def evaluate(c: _Cand):
        # refine each candidate to its own local bezel circle
        mcx, mcy, mr = c.cx, c.cy, c.r
        has_circle = (c.source == "hough")
        refined = _local_hough(roi_blur, mcx, mcy, r_min, r_max, cfg)
        if refined is not None and math.hypot(refined[0] - mcx,
                                              refined[1] - mcy) <= 1.5 * r_max:
            mcx, mcy, mr = refined
            has_circle = True
        dark = _darkness(roi_gray, roi_fg, mcx, mcy, mr, cfg)
        has_g, gpx = _glint(roi_gray, roi_fg, mcx, mcy, mr, cfg)
        circ = c.circ if c.source == "blob" else 0.0
        # weak prior: prefer the upper-outer part of the region (camera spot),
        # downweight the lower-inner corner where the lens edge lives
        prior = 1.0 - 0.25 * (abs(mcx - rx) / max(1.0, rx)
                              + abs(mcy - ry) / max(1.0, ry))
        sig = (0.30 * (1.0 if has_circle else 0.0)
               + 0.40 * (1.0 if has_g else 0.0)
               + 0.20 * dark
               + 0.10 * circ) * (0.85 + 0.15 * prior)
        return sig, (mcx, mcy, mr, has_circle, dark, has_g, gpx, circ)

    best = max((evaluate(c) for c in cands), key=lambda z: z[0])
    mcx, mcy, mr, has_circle, dark, has_g, gpx, circ = best[1]

    # final features from the chosen module --------------------------------
    out.circle = (roi.x + mcx, roi.y + mcy, mr)
    out.circ = round(circ, 3)
    out.f_circle = 1.0 if has_circle else 0.0
    out.f_dark = dark
    out.f_spec = 1.0 if has_g else 0.0
    out.spec_pixels = [(roi.x + px, roi.y + py) for px, py in gpx]
    if circ >= cfg.circ_min:
        out.f_blob = float(np.clip((circ - cfg.circ_min) / (1.0 - cfg.circ_min),
                                   0.0, 1.0))
    out.f_thick = _thickness(roi, seg, loc, cfg)
    return out


def _thickness(roi: CornerROI, seg: Segment, loc: Located, cfg: Config) -> float:
    """Max frame thickness in the corner band, normalised by frame width."""
    dist = cv2.distanceTransform(seg.frame_mask, cv2.DIST_L2, 5)
    band = dist[roi.y:roi.y + roi.h, roi.x:roi.x + roi.w]
    if band.size == 0:
        return 0.0
    # normalise by frame width so it is scale-invariant
    ratio = (2.0 * float(band.max())) / max(1.0, 0.5 * seg.bbox[2])
    return float(np.clip((ratio - cfg.thick_lo) / (cfg.thick_hi - cfg.thick_lo),
                         0.0, 1.0))


def extract(frames, seg: Segment, loc: Located, cfg: Config):
    """Return {'L': CornerFeatures, 'R': CornerFeatures}."""
    result = {}
    for roi in loc.rois:
        result[roi.side] = _corner_features(roi, frames, seg, loc, cfg)
    return result
