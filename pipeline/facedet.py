"""Stage 1.5 - Haar face detection, the localization anchor for WORN glasses.

Pure classical CV can't find glasses on a face (hair/clothing/background always
dominate the frame), so we anchor on the *face* instead: an OpenCV Haar cascade
(Viola-Jones, no neural net, ships in OpenCV + OpenCV.js). Frontal + profile
cascades, retried over a few in-plane rotations, detect the face in every
`ray_ban_face` sample. `locate` then derives the two top-outer corner ROIs from
this box; when no face is found we fall back to the held-glasses lens path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

# Bundled cascades (single source shared with the browser port). Fall back to
# OpenCV's install dir if the bundled copies are missing.
_HAAR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", "haar")


@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int
    method: str          # which cascade/rotation found it


_CASCADE_CACHE: dict = {}


def _cascade(name: str):
    if name not in _CASCADE_CACHE:
        path = os.path.join(_HAAR_DIR, name)
        if not os.path.exists(path):
            path = cv2.data.haarcascades + name
        _CASCADE_CACHE[name] = cv2.CascadeClassifier(path)
    return _CASCADE_CACHE[name]


# Upright only: the live-webcam use case holds the glasses/face roughly level, and
# adding rotated passes hallucinated faces on studio product shots (false META on
# normal glasses). This operating point (mn=6, no rotation, +profile) was tuned to
# give ZERO false faces on normal_frame while still catching 17/20 worn faces.
ROTATIONS = (0,)
_FRONTALS = ("haarcascade_frontalface_default.xml",
             "haarcascade_frontalface_alt2.xml")
_PROFILE = "haarcascade_profileface.xml"
_SCALE_FACTOR = 1.05
_MIN_NEIGHBORS = 6
_MIN_SIZE_FRAC = 0.10        # min face side as a fraction of min(H, W)


def _rot(gray, deg):
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    r = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=127)
    minv = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -deg, 1.0)
    return r, minv


def _map_box_back(box, minv):
    x, y, w, h = box
    pts = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]],
                   np.float32).reshape(-1, 1, 2)
    p = cv2.transform(pts, minv).reshape(-1, 2)
    x0, y0 = float(p[:, 0].min()), float(p[:, 1].min())
    x1, y1 = float(p[:, 0].max()), float(p[:, 1].max())
    return (int(round(x0)), int(round(y0)),
            int(round(x1 - x0)), int(round(y1 - y0)))


def detect_face(gray: np.ndarray) -> Face | None:
    """Largest plausible face box, or None. gray = canonical raw grayscale."""
    H, W = gray.shape[:2]
    min_side = int(_MIN_SIZE_FRAC * min(H, W))
    ms = (min_side, min_side)

    best = None                      # (area, box, method)

    def consider(box, method):
        nonlocal best
        area = box[2] * box[3]
        if best is None or area > best[0]:
            best = (area, box, method)

    # frontal cascades, upright first then rotated
    for deg in ROTATIONS:
        g, minv = _rot(gray, deg) if deg else (gray, None)
        for name in _FRONTALS:
            found = _cascade(name).detectMultiScale(g, _SCALE_FACTOR,
                                                    _MIN_NEIGHBORS, minSize=ms)
            for b in found:
                bb = _map_box_back(b, minv) if deg else tuple(int(v) for v in b)
                consider(bb, f"{name.split('_')[1]}@{deg}")
        if best is not None and deg == 0:
            break                    # good upright frontal -> stop early

    # profile fallback (image + mirror) only if nothing found yet
    if best is None:
        prof = _cascade(_PROFILE)
        for g, tag in ((gray, "profileL"), (cv2.flip(gray, 1), "profileR")):
            for b in prof.detectMultiScale(g, _SCALE_FACTOR, _MIN_NEIGHBORS,
                                            minSize=ms):
                x, y, w, h = (int(v) for v in b)
                if tag == "profileR":
                    x = W - x - w
                consider((x, y, w, h), tag)

    if best is None:
        return None
    _, (x, y, w, h), method = best
    return Face(x=x, y=y, w=w, h=h, method=method)
