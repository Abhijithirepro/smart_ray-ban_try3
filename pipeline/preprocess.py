"""Stage 1 - load, canonical resize, grayscale variants.

Produces a small bundle of working images that the rest of the pipeline shares:
  - color    : BGR, resized to canonical width
  - gray     : raw grayscale
  - gray_blur: Gaussian-blurred gray (Hough input)
  - gray_eq  : CLAHE-equalised gray (thresholding / contrast on black frames)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import Config


@dataclass
class Frames:
    color: np.ndarray
    gray: np.ndarray
    gray_blur: np.ndarray
    gray_eq: np.ndarray
    scale: float          # resize factor applied (new_w / orig_w)
    orig_shape: tuple     # (h, w) of the original image


def load_image(path: str) -> np.ndarray:
    """Load as BGR. Falls back to Pillow if OpenCV can't decode (e.g. some PNGs)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        raise FileNotFoundError(f"Could not read image: {path}")
    pil = Image.open(path).convert("RGB")
    return np.asarray(pil)[:, :, ::-1].copy()  # RGB -> BGR


def _flatten_alpha(img: np.ndarray) -> np.ndarray:
    """Composite an RGBA/transparent image onto white (product PNGs are RGBA)."""
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        alpha = (img[:, :, 3:4].astype(np.float32)) / 255.0
        white = np.full_like(bgr, 255.0)
        out = bgr * alpha + white * (1.0 - alpha)
        return out.astype(np.uint8)
    return img


def preprocess(path: str, cfg: Config) -> Frames:
    raw = load_image(path)
    raw = _flatten_alpha(raw)
    orig_h, orig_w = raw.shape[:2]

    scale = cfg.target_width / float(orig_w)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    color = cv2.resize(raw, (cfg.target_width, max(1, round(orig_h * scale))),
                       interpolation=interp)

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    k = cfg.gauss_ksize | 1  # force odd
    gray_blur = cv2.GaussianBlur(gray, (k, k), 0)

    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                            tileGridSize=(cfg.clahe_tile, cfg.clahe_tile))
    gray_eq = clahe.apply(gray)

    return Frames(color=color, gray=gray, gray_blur=gray_blur, gray_eq=gray_eq,
                  scale=scale, orig_shape=(orig_h, orig_w))
