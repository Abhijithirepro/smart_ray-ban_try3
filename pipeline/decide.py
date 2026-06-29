"""Stage 5 - per-corner scoring, fusion, verdict.

Single-camera target: the latest Ray-Ban Meta has ONE camera (one corner), so
the primary rule is "fire if EITHER corner looks like a camera". The opposite
corner being empty (or carrying only a tiny privacy LED) is expected and NOT
required. A two-camera Ray-Ban Stories simply makes both corners fire - still
META. The privacy LED is never used as a feature.
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


def _passes_gate(f: CornerFeatures, cfg: Config) -> bool:
    """Chunkiness gate: a real module sits in a chunky corner, UNLESS we have an
    unmistakable circle+specular pair (which bypasses the gate)."""
    if not cfg.use_thickness_gate:
        return True
    if f.f_circle >= 1.0 and f.f_spec >= 1.0:
        return True
    return f.f_thick > 0.0


def decide(features: dict, cfg: Config, seg=None, shape=None) -> Verdict:
    fL, fR = features["L"], features["R"]
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

    fire = {}
    for side, f, s in (("L", fL, sL), ("R", fR, sR)):
        fire[side] = (s >= cfg.corner_thresh) and _passes_gate(f, cfg)

    if cfg.require_both:
        is_meta = fire["L"] and fire["R"] and min(sL, sR) >= cfg.strong_thresh
    else:
        # single-corner fire is sufficient
        best_side = "L" if sL >= sR else "R"
        is_meta = fire[best_side] and max(sL, sR) >= cfg.strong_thresh

    overall = max(sL, sR)
    fired = None
    if is_meta:
        if cfg.require_both:
            fired = "L+R"
        else:
            fired = "L" if sL >= sR else "R"

    if is_meta:
        reason = f"camera module detected in corner {fired}"
    elif max(sL, sR) >= cfg.corner_thresh:
        reason = "corner candidate too weak / failed chunkiness gate"
    else:
        reason = "no camera-like module in either corner"

    return Verdict(verdict="META" if is_meta else "NORMAL",
                   overall_score=round(overall, 3),
                   score_left=round(sL, 3),
                   score_right=round(sR, 3),
                   fired_corner=fired,
                   reason=reason)
