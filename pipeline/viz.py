"""Stage 6 - debug overlay + JSON dump.

Annotated overlay colour key:
  blue   frame bbox
  green  lens boxes / centres
  yellow corner ROIs, each labelled with its learned P(camera)
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from pipeline.decide import Verdict
from pipeline.locate import Located
from pipeline.segment import Segment


BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)


def _i(v):
    return int(round(v))


def annotate(frames, seg: Segment, loc: Located, feats: dict,
             verdict: Verdict) -> np.ndarray:
    img = frames.color.copy()

    if loc.path == "face" and loc.face is not None:
        f = loc.face
        cv2.rectangle(img, (f.x, f.y), (f.x + f.w, f.y + f.h), BLUE, 2)
    else:
        x, y, w, h = seg.bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), BLUE, 2)
        for lens in (loc.lens_left, loc.lens_right):
            if lens is None:
                continue
            cv2.rectangle(img, (_i(lens.cx - lens.hw), _i(lens.cy - lens.hh)),
                          (_i(lens.cx + lens.hw), _i(lens.cy + lens.hh)), GREEN, 1)
            cv2.circle(img, (_i(lens.cx), _i(lens.cy)), 3, GREEN, -1)

    for roi in loc.rois:
        cv2.rectangle(img, (roi.x, roi.y),
                      (roi.x + roi.w, roi.y + roi.h), YELLOW, 2)
        f = feats[roi.side]
        cam = "" if f.cam_prob is None else f"  Pcam={f.cam_prob:.2f}"
        cv2.putText(img, f"{roi.side}{cam}", (roi.x, max(12, roi.y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)

    # header banner
    color = RED if verdict.verdict == "META" else GREEN
    pL, pR = feats["L"].cam_prob, feats["R"].cam_prob
    if pL is not None and pR is not None:
        banner = f"{verdict.verdict}  Pcam L={pL:.2f} R={pR:.2f}"
    else:
        banner = f"{verdict.verdict}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, banner, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                cv2.LINE_AA)
    return img


def _zoom(frames, roi, factor=4):
    if roi.w < 2 or roi.h < 2:
        return np.zeros((40, 40, 3), np.uint8)
    crop = frames.color[roi.y:roi.y + roi.h, roi.x:roi.x + roi.w]
    return cv2.resize(crop, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_NEAREST)


def montage(frames, loc: Located) -> np.ndarray:
    """ROI zooms side by side, upscaled so tiny modules are visible."""
    tiles = [_zoom(frames, roi) for roi in loc.rois]
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 8,
                                cv2.BORDER_CONSTANT, value=(40, 40, 40))
             for t in tiles]
    return np.hstack(tiles)


def features_to_dict(feats: dict, loc: Located, seg: Segment,
                     verdict: Verdict) -> dict:
    return {
        "verdict": verdict.verdict,
        "prob_left": verdict.prob_left,
        "prob_right": verdict.prob_right,
        "fired_corner": verdict.fired_corner,
        "reason": verdict.reason,
        "path": loc.path,
        "r_lens": round(loc.r_lens, 2),
        "bbox": list(seg.bbox),
        "face": ([loc.face.x, loc.face.y, loc.face.w, loc.face.h]
                 if loc.face is not None else None),
        "segment_method": seg.method,
        "locate_method": loc.method,
        "lens_centers": (
            {"L": [round(loc.lens_left.cx, 1), round(loc.lens_left.cy, 1)],
             "R": [round(loc.lens_right.cx, 1), round(loc.lens_right.cy, 1)]}
            if loc.lens_left is not None and loc.lens_right is not None else None),
        "per_corner": {"L": {"cam_prob": feats["L"].cam_prob},
                       "R": {"cam_prob": feats["R"].cam_prob}},
    }


def save_debug(path: str, frames, seg, loc, feats, verdict, debug_dir: str):
    os.makedirs(debug_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]

    overlay = annotate(frames, seg, loc, feats, verdict)
    mont = montage(frames, loc)
    # pad overlay/montage to same width and stack vertically
    W = max(overlay.shape[1], mont.shape[1])
    overlay = cv2.copyMakeBorder(overlay, 0, 0, 0, W - overlay.shape[1],
                                 cv2.BORDER_CONSTANT, value=(40, 40, 40))
    mont = cv2.copyMakeBorder(mont, 0, 0, 0, W - mont.shape[1],
                              cv2.BORDER_CONSTANT, value=(40, 40, 40))
    combined = np.vstack([overlay, mont])

    img_path = os.path.join(debug_dir, f"{stem}_debug.png")
    json_path = os.path.join(debug_dir, f"{stem}_features.json")
    cv2.imwrite(img_path, combined)
    with open(json_path, "w") as fh:
        json.dump(features_to_dict(feats, loc, seg, verdict), fh, indent=2)
    return img_path, json_path
