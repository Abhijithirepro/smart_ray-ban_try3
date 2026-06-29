"""Stage 2 - find the frame bounding box + a filled frame mask.

Primary path: inverse-Otsu threshold (dark frame on light background becomes
foreground) -> morphological clean -> largest external contour -> bbox.
A sanity gate rejects implausible bboxes; on failure we fall back to a Canny
edge contour, then to a central crop.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import Config


@dataclass
class Segment:
    bbox: tuple              # (x, y, w, h) in canonical pixels
    frame_mask: np.ndarray   # uint8 {0,255}, filled, full canonical size
    method: str              # "otsu" | "canny" | "manual" | "center-crop"
    valid: bool = True       # did we isolate a plausible glasses frame?

    def isolated(self, cfg, shape) -> bool:
        """An isolated-glasses frame (the operating condition) vs a fallback /
        whole-image bbox (e.g. a face photo where segmentation collapsed)."""
        if not self.valid or self.method == "center-crop":
            return False
        H, W = shape[:2]
        area_frac = (self.bbox[2] * self.bbox[3]) / float(W * H)
        return area_frac <= cfg.seg_isolated_max


def _largest_contour(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def _bbox_ok(bbox, shape, cfg: Config) -> bool:
    x, y, w, h = bbox
    H, W = shape[:2]
    area_frac = (w * h) / float(W * H)
    if not (cfg.seg_min_area_frac < area_frac < cfg.seg_max_area_frac):
        return False
    aspect = w / float(h) if h else 0.0
    return cfg.seg_min_aspect <= aspect <= cfg.seg_max_aspect


def _filled_mask_from_contour(cnt, shape) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    return mask


def segment(gray_eq: np.ndarray, cfg: Config,
            manual_bbox: tuple | None = None) -> Segment:
    H, W = gray_eq.shape[:2]

    if manual_bbox is not None:
        x, y, w, h = manual_bbox
        mask = np.zeros((H, W), np.uint8)
        mask[y:y + h, x:x + w] = 255
        return Segment(bbox=manual_bbox, frame_mask=mask, method="manual")

    # ---- primary: inverse Otsu -------------------------------------------
    _, th = cv2.threshold(gray_eq, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ck = cfg.seg_close_ksize | 1
    ok = cfg.seg_open_ksize | 1
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          np.ones((ck, ck), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                          np.ones((ok, ok), np.uint8))
    cnt = _largest_contour(th)
    if cnt is not None:
        bbox = cv2.boundingRect(cnt)
        if _bbox_ok(bbox, gray_eq.shape, cfg):
            return Segment(bbox=bbox,
                           frame_mask=_filled_mask_from_contour(cnt, gray_eq.shape),
                           method="otsu")

    # ---- fallback: Canny edges -------------------------------------------
    edges = cv2.Canny(gray_eq, cfg.canny_lo, cfg.canny_hi)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    cnt = _largest_contour(edges)
    if cnt is not None:
        bbox = cv2.boundingRect(cnt)
        if _bbox_ok(bbox, gray_eq.shape, cfg):
            return Segment(bbox=bbox,
                           frame_mask=_filled_mask_from_contour(cnt, gray_eq.shape),
                           method="canny")

    # ---- last resort: central 90% crop -----------------------------------
    x = int(0.05 * W)
    y = int(0.05 * H)
    w = int(0.90 * W)
    h = int(0.90 * H)
    mask = np.zeros((H, W), np.uint8)
    mask[y:y + h, x:x + w] = 255
    return Segment(bbox=(x, y, w, h), frame_mask=mask, method="center-crop",
                   valid=False)
