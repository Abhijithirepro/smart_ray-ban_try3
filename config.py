"""Central configuration for the Meta-glasses detector.

Every tunable lives here as a single @dataclass so the whole pipeline can be
re-tuned from one place (or from a JSON override file via --config). All spatial
parameters are expressed *relative* to the detected lens radius (R_lens) or the
frame width, so the detector is scale-invariant after the canonical resize.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace


@dataclass
class Config:
    # ---- preprocess -------------------------------------------------------
    target_width: int = 1000           # canonical width after resize
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    gauss_ksize: int = 5               # blur kernel for Hough input

    # ---- segment (frame bbox) --------------------------------------------
    seg_close_ksize: int = 7
    seg_open_ksize: int = 3
    seg_min_area_frac: float = 0.10    # bbox area / image area lower bound
    seg_max_area_frac: float = 0.97    # upper bound
    seg_min_aspect: float = 1.4        # glasses are wide (w/h)
    seg_max_aspect: float = 4.5
    seg_isolated_max: float = 0.80     # bbox area frac above this = not isolated
                                       # (a face filling the frame, not glasses)
    canny_lo: int = 50
    canny_hi: int = 150

    # ---- locate (top-outer search region) --------------------------------
    # The camera sits OUTSIDE the lens, on the end piece. So the search ROI runs
    # from the lens's OUTER edge outward to the frame edge, across the top - i.e.
    # it begins where the green lens box ends. Anchoring to the lens makes it
    # scale with the frame (near/far) and follow the actual eye shape.
    roi_lens_overlap: float = 0.15     # overlap back into the lens (* lens half-width)
    roi_y_below: float = 0.15          # extend below the lens centre (* lens half-height)
    roi_min_w_frac: float = 0.10       # floor on ROI width (* frame width), safety

    # ---- features: radius window for a camera module ---------------------
    # Based on FRAME WIDTH (stable), not R_lens. Module radius ~0.02 * width.
    cam_r_min_w: float = 0.014         # * frame width
    cam_r_max_w: float = 0.037         # * frame width
    cam_r_min_px: int = 4

    # (a) small-circle Hough
    hc_dp: float = 1.0
    hc_param1: int = 100
    hc_param2: int = 11                # primary sensitivity dial (lower = more)
    hc_min_dist_frac: float = 0.4      # * roi width

    # glint-seeded candidates (camera's most reliable cue on black frames)
    glint_seed_thr: int = 150          # min brightness on-frame to seed a glint
    glint_seed_below_max: int = 45     # OR within this of the on-frame max
    glint_seed_max_area: int = 80      # a glint is tiny; reject big highlights
    hc_param2_local: int = 9           # local refine Hough (more sensitive)

    # (b) dark blob + circularity
    dark_k: float = 0.6                # T = mean - dark_k * std  (relative thresh)
    circ_min: float = 0.62             # minimum circularity to accept a blob

    # (c) local darkness vs surrounding frame ring
    dark_norm: float = 40.0            # normaliser for (ring_median - blob_mean)

    # (d) specular highlight
    spec_abs_min: int = 230            # absolute brightness floor for a glint
    spec_rel_below_max: int = 25       # OR within this of roi.max()
    spec_min_px: int = 2               # need at least this many bright px
    spec_max_area_frac: float = 0.25   # but not more than this fraction of blob
    glint_central_frac: float = 0.6    # glint must lie within this * r of centre

    # (f) frame thickness (chunkiness) in the corner band
    thick_lo: float = 0.18             # ratio (2*distTransform/R_lens) -> 0
    thick_hi: float = 0.40             # ratio -> 1

    # ---- scoring weights (per corner) ------------------------------------
    w_circle: float = 0.25
    w_blob: float = 0.25
    w_dark: float = 0.20
    w_spec: float = 0.20
    w_thick: float = 0.10

    # ---- decision thresholds ---------------------------------------------
    corner_thresh: float = 0.50        # a corner "has a module" if score >= this
    strong_thresh: float = 0.60        # fire META if max(sL, sR) >= this
    use_thickness_gate: bool = True    # require chunky corner unless circle+spec
    require_both: bool = False         # if True, both corners must fire

    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load defaults and merge a JSON override file on top."""
        cfg = cls()
        with open(path, "r") as fh:
            overrides = json.load(fh)
        known = {f.name for f in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return replace(cfg, **{k: overrides[k] for k in overrides})
