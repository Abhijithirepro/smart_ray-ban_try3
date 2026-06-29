"""Tiny camera-vs-no-camera classifier on the corner-ROI crop.

Classical circle/darkness/glint cues cannot separate a Ray-Ban camera module
from a normal frame's rounded corner (both are "a dark circle in the corner").
That distinction is structural/semantic, so we learn it: a HOG descriptor over
the corner crop (which encodes the circular bezel of a real module vs the
L-shaped edge of a bare frame corner) feeds an L2-regularised logistic
regression. The model is intentionally tiny and self-contained (pure numpy +
OpenCV's HOG), with no external ML dependency.

Feature pipeline (identical at train and inference):
  corner ROI -> 32x32 gray -> R-corner flipped to canonical orientation
             -> per-crop illumination normalisation -> HOG -> standardise.
"""
from __future__ import annotations

import numpy as np
import cv2

CROP = 32
# winSize, blockSize, blockStride, cellSize, nbins  -> 144-dim descriptor
_HOG = cv2.HOGDescriptor((CROP, CROP), (16, 16), (16, 16), (8, 8), 9)


def corner_crop(frames, roi) -> np.ndarray:
    """Fixed-size, orientation-canonical grayscale crop of a corner ROI.

    The right corner is mirrored so both corners present their camera spot in
    the same (upper-outer-left) orientation; this halves what the model must
    learn and lets left/right share training examples.
    """
    g = frames.gray[roi.y:roi.y + roi.h, roi.x:roi.x + roi.w]
    if g.size == 0:
        return np.zeros((CROP, CROP), np.uint8)
    g = cv2.resize(g, (CROP, CROP), interpolation=cv2.INTER_AREA)
    if roi.side == "R":
        g = cv2.flip(g, 1)
    return g


def hog_feat(crop: np.ndarray) -> np.ndarray:
    """Illumination-normalise the crop, then return its HOG descriptor."""
    c = crop.astype(np.float32)
    c = (c - c.mean()) / (c.std() + 1e-6)
    c = np.clip(c * 64.0 + 128.0, 0, 255).astype(np.uint8)
    return _HOG.compute(c).ravel().astype(np.float32)


class CameraClf:
    """Standardise -> linear logit -> sigmoid. Weights persisted as .npz."""

    def __init__(self, w=None, b=0.0, mu=None, sd=None):
        self.w = w
        self.b = float(b)
        self.mu = mu
        self.sd = sd

    def prob(self, feat: np.ndarray) -> float:
        x = (feat - self.mu) / self.sd
        z = float(x @ self.w + self.b)
        return 1.0 / (1.0 + np.exp(-z))

    def prob_from_crop(self, crop: np.ndarray) -> float:
        return self.prob(hog_feat(crop))

    def save(self, path: str):
        np.savez(path, w=self.w, b=np.float32(self.b), mu=self.mu, sd=self.sd)

    @classmethod
    def load(cls, path: str) -> "CameraClf":
        d = np.load(path)
        return cls(d["w"].astype(np.float32), float(d["b"]),
                   d["mu"].astype(np.float32), d["sd"].astype(np.float32))


def train_logreg(X: np.ndarray, y: np.ndarray, l2=1.0, lr=0.1, iters=2000):
    """Standardise X, fit logistic regression by full-batch gradient descent.

    Returns (w, b, mu, sd). L2 regularisation (not applied to bias) keeps the
    144-dim model from memorising a 19-image training set.
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    Xn = (X - mu) / sd
    n, d = Xn.shape
    w = np.zeros(d, np.float64)
    b = 0.0
    for _ in range(iters):
        z = Xn @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        gw = Xn.T @ g / n + l2 * w / n
        gb = g.mean()
        w -= lr * gw
        b -= lr * gb
    return w.astype(np.float32), float(b), mu.astype(np.float32), sd.astype(np.float32)
