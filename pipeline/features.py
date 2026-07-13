"""Stage 4 - per-corner camera probability from the learned classifier.

The verdict is decided entirely by the learned corner classifier
(pipeline/camera_clf). For each top-outer corner ROI placed by locate, we take
its orientation-canonical crop and read off P(camera). No hand-tuned circle /
blob / darkness / glint measurements are used any more - that distinction is
structural/semantic, so we let the model make it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from config import Config
from pipeline.locate import Located
from pipeline.segment import Segment


@dataclass
class CornerFeatures:
    side: str
    cam_prob: float | None = None        # peak learned P(camera) over candidates
    best_roi: object | None = None       # the CornerROI that gave the peak
    n_candidates: int = 0


_CLF_CACHE = {}


def _load_clf(path: str):
    """Lazily load (and cache) the corner camera classifier; None if absent."""
    if path not in _CLF_CACHE:
        from pipeline import camera_clf
        _CLF_CACHE[path] = (camera_clf.CameraClf.load(path)
                            if os.path.exists(path) else None)
    return _CLF_CACHE[path]


def extract(frames, seg: Segment, loc: Located, cfg: Config):
    """Return {'L': CornerFeatures, 'R': CornerFeatures} with the PEAK learned
    P(camera) over each side's candidate ROIs (a local search that absorbs
    corner-placement imprecision, especially on the face-anchored path)."""
    clf = _load_clf(cfg.cam_clf_path)
    result = {}
    for side in ("L", "R"):
        cands = loc.candidates.get(side, [])
        f = CornerFeatures(side=side, n_candidates=len(cands))
        if clf is not None and cands:
            from pipeline import camera_clf
            best_p, best_roi = -1.0, None
            for roi in cands:
                p = clf.prob_from_crop(camera_clf.corner_crop(frames, roi))
                if p > best_p:
                    best_p, best_roi = p, roi
            f.cam_prob = round(best_p, 3)
            f.best_roi = best_roi
        result[side] = f
    # keep loc.rois pointing at the chosen crops so viz shows what was scored
    loc.rois = [result["L"].best_roi or loc.candidates["L"][0],
                result["R"].best_roi or loc.candidates["R"][0]]
    return result
