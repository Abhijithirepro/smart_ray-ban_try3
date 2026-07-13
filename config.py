"""Central configuration for the Meta-glasses detector.

Every tunable lives here as a single @dataclass so the whole pipeline can be
re-tuned from one place (or from a JSON override file via --config). All spatial
parameters are expressed *relative* to the frame width or the detected lens box,
so the detector is scale-invariant after the canonical resize.

The verdict is produced by the learned corner classifier (pipeline/camera_clf).
The earlier stages (preprocess -> segment -> locate) exist only to find the two
top-outer corner crops the classifier scores; there are no hand-tuned circle /
darkness / glint measurements any more.
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
    gauss_ksize: int = 5               # blur kernel for the canonical gray

    # ---- segment (frame bbox) --------------------------------------------
    seg_close_ksize: int = 7
    seg_open_ksize: int = 3
    seg_min_area_frac: float = 0.10    # bbox area / image area lower bound
    seg_max_area_frac: float = 0.97    # upper bound
    seg_min_aspect: float = 1.4        # glasses are wide (w/h)
    seg_max_aspect: float = 4.5
    canny_lo: int = 50
    canny_hi: int = 150

    # ---- worn case: Haar face anchor + top-outer corner search -----------
    # Pure CV can't find glasses on a face, so we anchor on the face (Haar) and
    # search a small grid of candidate corner crops per side, taking the peak
    # P(camera). All offsets are face-relative -> scale invariant.
    face_search_sizes: tuple = (0.28, 0.36, 0.45)   # module box side / face width
    face_search_dy: tuple = (0.18, 0.28, 0.38, 0.48)  # eye-line, * face height
    face_search_dx: tuple = (-0.05, 0.05, 0.15)     # inset from outer edge, * face w

    # ---- locate (top-outer corner ROI the classifier scores) -------------
    # The camera sits OUTSIDE the lens, on the end piece. So the search ROI runs
    # from the lens's OUTER edge outward to the frame edge, across the top - i.e.
    # it begins where the green lens box ends. Anchoring to the lens makes it
    # scale with the frame (near/far) and follow the actual eye shape.
    roi_lens_overlap: float = 0.15     # overlap back into the lens (* lens half-width)
    roi_y_below: float = 0.15          # extend below the lens centre (* lens half-height)
    roi_min_w_frac: float = 0.10       # floor on ROI width (* frame width), safety
    roi_outer_max: float = 1.0         # cap ROI outer edge at this * lens.hw beyond
                                       # the lens edge (keeps ROI off hands/hair)

    # ---- glasses-region crop (input to the CNN classifier) ---------------
    # One bbox around the whole glasses (frame + end pieces). Worn: face width
    # across the eye line; held/studio: padded segment bbox. Loose on purpose -
    # the CNN's RandomResizedCrop augmentation absorbs the slack.
    region_wpad: float = 0.06          # widen past the face box each side (* face w)
    region_up: float = 0.24            # above the eye line (* face h)
    region_down: float = 0.20          # below the eye line (* face h)
    region_seg_pad: float = 0.08       # pad the segment bbox (* bbox side)

    # ---- learned camera classifier (produces the verdict) ----------------
    cam_clf_path: str = "models/camera_clf.npz"  # corner-crop classifier weights
    cam_clf_thresh: float = 0.50       # per-corner P(camera) needed to fire

    # ---- top-outer corner crop (input to the module CNN) -----------------
    # The Ray-Ban camera module lives at the top-outer end piece. These cut a
    # square corner from the ORIGINAL full-res image (module stays resolvable),
    # face-relative so scale-invariant. corner_top: box top as a fraction of face
    # height below the face top; corner_out: how far outboard of the face box edge
    # to start (temples extend past the face box).
    corner_size: float = 0.42          # box side / face width
    corner_yc: float = 0.05            # box centre above the eye line (* face h)
    corner_out: float = 0.06           # outward past the face edge (* face w)
    corner_input: int = 112            # CNN input side for a corner crop
    corner_clf_thresh: float = 0.50    # per-corner P(module); both must pass

    # ---- corner candidate GRID (localization robustness) -----------------
    # A single fixed box slides off the module on tilted / oddly-framed heads. We
    # scan a small grid of candidate boxes per side and take the max P(module).
    # All offsets are face-relative -> scale invariant. Kept modest so a normal
    # frame corner doesn't get many spurious chances (that is countered by
    # hard-negative training on these same grid positions). Fractions, not pixels.
    corner_grid_sizes: tuple = (0.36, 0.42, 0.52)      # box side / face width
    corner_grid_yc: tuple = (-0.02, 0.05, 0.14)        # centre above eye line (* face h)
    corner_grid_out: tuple = (0.0, 0.12)               # outward past face edge (* face w)

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
