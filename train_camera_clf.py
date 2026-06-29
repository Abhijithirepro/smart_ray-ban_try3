#!/usr/bin/env python3
"""Train the corner-ROI camera classifier (see pipeline/camera_clf.py).

Positives: ray_ban_frame/  (both corners carry a camera module)
Negatives: normal_frame/    (neither corner has a camera)

Each image yields one crop per corner. We report LEAVE-ONE-IMAGE-OUT CV first
(grouped by image so a single frame's two corners never straddle train/test) -
this is the honest generalisation estimate on a tiny set. The shipped model is
then retrained on all images. Augmentation (brightness / shift / rotate) is
applied to TRAIN crops only, never to held-out test crops.

    python3 train_camera_clf.py
"""
import glob
import os

import cv2
import numpy as np

from config import Config
from pipeline import preprocess, segment, locate, camera_clf as CC

POS = "ray_ban_frame"   # label 1
NEG = "normal_frame"    # label 0
MODEL_PATH = "models/camera_clf.npz"


def files(folder):
    return sorted(f for f in glob.glob(f"{folder}/*")
                  if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))


def crops_for_image(path, cfg):
    """Return list of (crop, side) for an image's corner ROIs."""
    fr = preprocess.preprocess(path, cfg)
    seg = segment.segment(fr.gray_eq, cfg)
    loc = locate.locate(fr.gray_eq, seg, cfg)
    return [CC.corner_crop(fr, roi) for roi in loc.rois]


def augment(crop):
    """Small label-preserving variants of a canonical crop."""
    out = [crop]
    h, w = crop.shape
    for scale in (0.82, 1.18):                       # brightness
        out.append(np.clip(crop.astype(np.float32) * scale, 0, 255).astype(np.uint8))
    for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):  # shift
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        out.append(cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REPLICATE))
    for ang in (-6, 6):                               # rotate
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        out.append(cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REPLICATE))
    return out


def build_dataset(cfg):
    """Return per-image records: list of (crops:list, label:int, name:str)."""
    data = []
    for folder, label in ((POS, 1), (NEG, 0)):
        for f in files(folder):
            data.append((crops_for_image(f, cfg), label, os.path.basename(f)))
    return data


def _feats(crops):
    return np.array([CC.hog_feat(c) for c in crops], np.float32)


def main():
    cfg = Config()
    data = build_dataset(cfg)
    npos = sum(1 for _, l, _ in data if l == 1)
    print(f"images: {npos} camera (+{2*npos} crops), "
          f"{len(data)-npos} no-camera (+{2*(len(data)-npos)} crops)")

    # ---- leave-one-image-out CV (honest generalisation estimate) ----
    correct = total = 0
    misses = []
    for i, (crops_i, yi, name) in enumerate(data):
        Xtr, ytr = [], []
        for j, (crops_j, yj, _) in enumerate(data):
            if j == i:
                continue
            for c in crops_j:
                for a in augment(c):
                    Xtr.append(CC.hog_feat(a)); ytr.append(yj)
        w, b, mu, sd = CC.train_logreg(np.array(Xtr, np.float32),
                                       np.array(ytr, np.float64))
        clf = CC.CameraClf(w, b, mu, sd)
        probs = [clf.prob(CC.hog_feat(c)) for c in crops_i]
        # image is "camera" iff BOTH corners predicted camera (the real rule)
        pred = 1 if all(p >= 0.5 for p in probs) else 0
        ok = pred == yi
        correct += ok; total += 1
        if not ok:
            misses.append((name, yi, [round(p, 2) for p in probs]))
    print(f"\nLEAVE-ONE-IMAGE-OUT CV (both-corners rule): "
          f"{correct}/{total} = {100*correct/total:.0f}%")
    for name, y, ps in misses:
        kind = "camera->missed" if y == 1 else "normal->FALSE POS"
        print(f"  MISS [{kind}] {name}  corner_probs={ps}")

    # ---- final model on ALL images ----
    Xall, yall = [], []
    for crops, y, _ in data:
        for c in crops:
            for a in augment(c):
                Xall.append(CC.hog_feat(a)); yall.append(y)
    w, b, mu, sd = CC.train_logreg(np.array(Xall, np.float32),
                                   np.array(yall, np.float64))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    CC.CameraClf(w, b, mu, sd).save(MODEL_PATH)
    print(f"\nsaved final model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
