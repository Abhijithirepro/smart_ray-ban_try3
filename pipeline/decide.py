"""Stage 5 - per-corner verdict.

Decision rule: a Ray-Ban Meta carries a circular camera module in BOTH top-outer
corners, so we fire META iff a Hough circle candidate (L1 *and* R1, the blue
circles in the debug overlay) is found in BOTH corner ROIs. The per-corner
weighted scores are still computed, but only for debug display - they no longer
drive the verdict. The privacy LED is never used as a feature.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import Config
from pipeline.features import CornerFeatures


@dataclass
class Verdict:
    verdict: str          # "META" | "NORMAL"
    overall_score: float
    score_left: float
    score_right: float
    fired_corner: str | None
    reason: str


def _corner_score(f: CornerFeatures, cfg: Config) -> float:
    return (cfg.w_circle * f.f_circle +
            cfg.w_blob * f.f_blob +
            cfg.w_dark * f.f_dark +
            cfg.w_spec * f.f_spec +
            cfg.w_thick * f.f_thick)


def decide(features: dict, cfg: Config, seg=None, shape=None) -> Verdict:
    fL, fR = features["L"], features["R"]
    # scores are debug-only now; they no longer drive the verdict
    sL = _corner_score(fL, cfg)
    sR = _corner_score(fR, cfg)

    # Domain gate: we only claim META when an isolated glasses frame was found.
    # If segmentation collapsed (e.g. a face photo, not a clean glasses shot),
    # the "corners" are meaningless and an eye can mimic a camera, so abstain.
    if seg is not None and shape is not None and not seg.isolated(cfg, shape):
        return Verdict(verdict="NORMAL", overall_score=round(max(sL, sR), 3),
                       score_left=round(sL, 3), score_right=round(sR, 3),
                       fired_corner=None,
                       reason="no isolated glasses frame found (out of operating "
                              "condition: needs a clean photo of just the glasses)")

    # Primary rule: a camera module must be present in BOTH corners.
    # When the learned classifier is available we use P(camera) per corner
    # (a Hough circle alone can't tell a camera from a rounded frame corner);
    # otherwise we fall back to raw Hough-candidate presence (L1 / R1).
    if fL.cam_prob is not None and fR.cam_prob is not None:
        has_L = fL.cam_prob >= cfg.cam_clf_thresh
        has_R = fR.cam_prob >= cfg.cam_clf_thresh
        cue = f"P(cam) L={fL.cam_prob:.2f} R={fR.cam_prob:.2f}"
    else:
        has_L = len(fL.hough_candidates) > 0
        has_R = len(fR.hough_candidates) > 0
        cue = "Hough circle (L1 / R1)"

    is_meta = has_L and has_R
    if is_meta:
        reason = f"camera in both corners [{cue}]"
        fired = "L+R"
    elif has_L or has_R:
        only = "L" if has_L else "R"
        reason = f"camera only in corner {only}, need both [{cue}]"
        fired = None
    else:
        reason = f"no camera in either corner [{cue}]"
        fired = None

    return Verdict(verdict="META" if is_meta else "NORMAL",
                   overall_score=round(max(sL, sR), 3),
                   score_left=round(sL, 3),
                   score_right=round(sR, 3),
                   fired_corner=fired,
                   reason=reason)
