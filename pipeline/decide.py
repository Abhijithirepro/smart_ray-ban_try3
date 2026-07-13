"""Stage 5 - verdict from the learned per-corner camera probability.

A Ray-Ban Meta carries a circular camera module in BOTH top-outer corners, so we
fire META iff the classifier reports P(camera) >= cam_clf_thresh in BOTH corners.
A domain gate first confirms we are actually looking at a glasses frame: the face
path is gated by "a face was found", the held path by a plausible two-lens frame
(see locate.two_lens_gate). The gate no longer rejects faces - detecting Meta
worn on a face is a goal - so the META/NORMAL split rests on the classifier.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import Config


@dataclass
class Verdict:
    verdict: str          # "META" | "NORMAL"
    prob_left: float | None
    prob_right: float | None
    fired_corner: str | None
    reason: str


def decide(features: dict, cfg: Config, gate_ok: bool = True,
           gate_reason: str = "") -> Verdict:
    fL, fR = features["L"], features["R"]
    pL, pR = fL.cam_prob, fR.cam_prob

    if not gate_ok:
        return Verdict(verdict="NORMAL", prob_left=pL, prob_right=pR,
                       fired_corner=None,
                       reason=f"no glasses frame found ({gate_reason})")

    if pL is None or pR is None:
        raise RuntimeError(
            "camera classifier unavailable: expected weights at "
            f"{cfg.cam_clf_path} (train with train_camera_clf.py)")

    has_L = pL >= cfg.cam_clf_thresh
    has_R = pR >= cfg.cam_clf_thresh
    cue = f"P(cam) L={pL:.2f} R={pR:.2f}"

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
                   prob_left=pL, prob_right=pR,
                   fired_corner=fired, reason=reason)
