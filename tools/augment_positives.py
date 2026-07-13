#!/usr/bin/env python3
"""Failure-mode-targeted offline augmentation for the Ray-Ban detector.

We only have ~68 train positives. Each augmentation here attacks a specific failure
cause measured in the old system (see the plan / detection_accuracy.csv):

  A. Occlusion (hair/hand over one temple, head turned) -> custom edge occlusion
     patch + Affine rotate/shear. The whole-glasses box stays; the object is still
     present, just partly hidden, so the detector learns to fire on partial views.
  B. Low resolution / 1-2 m distance -> Downscale + JPEG recompression + blur, which
     shrinks the camera module to a few pixels like the real 335px false-negatives.
  C. Clear / low-contrast frames & lighting -> brightness/contrast/gamma/hue + CLAHE.
  FP-hardening -> synthetic specular GLARE pasted onto NEGATIVE images near the lens
     corners, so "bright glint at a thick endpiece" is learned as NOT-a-module.

Reads data/boxes/boxes.json (the reviewed manifest), augments the TRAIN split only
(val stays 100% real for honest metrics), writes images under data/boxes/aug/ and a
manifest data/boxes/aug.json. export_dataset.py merges boxes.json + aug.json.

    python tools/augment_positives.py                          # defaults
    python tools/augment_positives.py --per-image 8 --neg-glare-frac 0.35
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import albumentations as A

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOXES_DIR = os.path.join(_REPO, "data", "boxes")
MANIFEST = os.path.join(BOXES_DIR, "boxes.json")
AUG_MANIFEST = os.path.join(BOXES_DIR, "aug.json")
AUG_IMG_DIR = os.path.join(BOXES_DIR, "aug")


def build_pipeline() -> A.Compose:
    """Photometric + geometric + quality transforms (albumentations 2.x).

    Occlusion (A) and glare (FP) are done separately in numpy so we control exactly
    where they land relative to the box; everything else is stock albumentations."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),                       # module is on both sides
            A.Affine(scale=(0.9, 1.12), rotate=(-12, 12), shear=(-8, 8),
                     translate_percent=(0.0, 0.06), p=0.85),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25,
                                 val_shift_limit=15, p=0.4),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.OneOf([A.CLAHE(clip_limit=2.0), A.Equalize()], p=0.25),
            # ---- cause B: distance / low-res webcam ----
            A.Downscale(scale_range=(0.28, 0.6), p=0.45),
            A.OneOf([A.MotionBlur(blur_limit=(3, 9)),
                     A.GaussianBlur(blur_limit=(3, 7)),
                     A.MedianBlur(blur_limit=5)], p=0.4),
            A.ImageCompression(quality_range=(30, 70), p=0.5),
            A.GaussNoise(std_range=(0.03, 0.12), p=0.3),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["cls"],
                                 min_visibility=0.35, min_area=64),
        seed=0,
    )


def build_neg_pipeline() -> A.Compose:
    """Same photometric/geometric/quality transforms as build_pipeline() but with NO
    bbox handling — negatives (normal glasses) carry no box. Used to multiply the
    negative set so worn normal glasses aren't drowned out by the x8 positives (the
    cause of the 'normal glasses read as Ray-Ban' false positives)."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.12), rotate=(-12, 12), shear=(-8, 8),
                     translate_percent=(0.0, 0.06), p=0.85),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25,
                                 val_shift_limit=15, p=0.4),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.OneOf([A.CLAHE(clip_limit=2.0), A.Equalize()], p=0.25),
            A.Downscale(scale_range=(0.28, 0.6), p=0.45),
            A.OneOf([A.MotionBlur(blur_limit=(3, 9)),
                     A.GaussianBlur(blur_limit=(3, 7)),
                     A.MedianBlur(blur_limit=5)], p=0.4),
            A.ImageCompression(quality_range=(30, 70), p=0.5),
            A.GaussNoise(std_range=(0.03, 0.12), p=0.3),
        ],
        seed=0,
    )


def add_edge_occlusion(img: np.ndarray, box_xywh, rng: np.random.Generator) -> np.ndarray:
    """Cause A: hide one temple/side of the glasses with a hair-like patch.

    Samples a texture strip from just above the box (usually hair/background) and
    pastes it over one vertical edge of the box with a feathered mask, so the object
    is partly occluded but the box label stays valid."""
    out = img.copy()
    H, W = out.shape[:2]
    x, y, w, h = [int(round(v)) for v in box_xywh]
    if w < 10 or h < 10:
        return out
    frac = rng.uniform(0.22, 0.42)          # how much of the box width to cover
    pw = max(6, int(w * frac))
    side = rng.choice(["L", "R"])
    px = x if side == "L" else x + w - pw
    py = y - int(0.05 * h)
    ph = int(h * rng.uniform(0.85, 1.1))
    px = max(0, min(px, W - 1)); py = max(0, min(py, H - 1))
    pw = min(pw, W - px); ph = min(ph, H - py)
    if pw < 4 or ph < 4:
        return out

    # texture source: a strip above the box (hair), else a dark fill
    src_y = max(0, y - ph)
    strip = out[src_y:src_y + ph, px:px + pw]
    if strip.shape[:2] != (ph, pw) or strip.size == 0:
        patch = np.zeros((ph, pw, 3), np.uint8)
        patch[:] = rng.integers(10, 60, size=3)
    else:
        patch = cv2.GaussianBlur(strip, (0, 0), 3)

    # feathered mask (soft inner edge so it doesn't look like a hard rectangle)
    mask = np.ones((ph, pw), np.float32)
    feather = max(4, pw // 4)
    ramp = np.linspace(0, 1, feather)
    if side == "L":
        mask[:, pw - feather:] *= ramp[::-1]
    else:
        mask[:, :feather] *= ramp
    mask = cv2.GaussianBlur(mask, (0, 0), 2)[..., None]
    region = out[py:py + ph, px:px + pw].astype(np.float32)
    out[py:py + ph, px:px + pw] = (patch * mask + region * (1 - mask)).astype(np.uint8)
    return out


def add_glare(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """FP-hardening: paste 1-3 bright specular blobs in the upper-middle band (where
    lenses/endpieces sit). Applied to NEGATIVES so a corner glint alone never means
    Ray-Ban."""
    out = img.astype(np.float32)
    H, W = out.shape[:2]
    layer = np.zeros((H, W), np.float32)
    for _ in range(rng.integers(1, 4)):
        cx = int(rng.uniform(0.15, 0.85) * W)
        cy = int(rng.uniform(0.20, 0.55) * H)
        ax = int(rng.uniform(0.01, 0.05) * W) + 3
        ay = int(rng.uniform(0.01, 0.04) * H) + 3
        cv2.ellipse(layer, (cx, cy), (ax, ay),
                    float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), max(ax, ay) * 0.5 + 2)
    tint = rng.choice([(255, 255, 255), (200, 255, 220), (220, 240, 255)])
    strength = rng.uniform(0.5, 0.95)
    for c in range(3):
        out[..., c] += layer * tint[c] * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-image", type=int, default=8,
                    help="augmented variants per train positive (default 8)")
    ap.add_argument("--occlusion-p", type=float, default=0.5,
                    help="prob. of adding an edge-occlusion patch to a positive variant")
    ap.add_argument("--neg-per-image", type=int, default=3,
                    help="augmented variants per train NEGATIVE (default 3) — "
                         "rebalances the class ratio so normal glasses aren't "
                         "outnumbered by the x8 positives (fixes false positives)")
    ap.add_argument("--neg-glare-frac", type=float, default=0.3,
                    help="fraction of train negatives to ALSO emit a glare-hardened copy of")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if not os.path.isfile(MANIFEST):
        raise SystemExit(f"run tools/bootstrap_boxes.py first (missing {MANIFEST})")
    manifest = json.load(open(MANIFEST))
    rng = np.random.default_rng(args.seed)
    tf = build_pipeline()
    neg_tf = build_neg_pipeline()
    os.makedirs(AUG_IMG_DIR, exist_ok=True)

    aug_records = []
    n_pos_in = n_pos_out = n_neg_out = n_drop = 0

    for r in manifest["images"]:
        if r["split"] != "train":
            continue
        src_path = os.path.join(BOXES_DIR, r["image"])
        img = cv2.imread(src_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        base = os.path.splitext(os.path.basename(r["image"]))[0]

        if r["label"] == 1 and r.get("box"):
            n_pos_in += 1
            x, y, w, h = r["box"]
            voc = [[x, y, x + w, y + h]]
            for k in range(args.per_image):
                res = tf(image=img, bboxes=voc, cls=[0])
                if not len(res["bboxes"]):
                    n_drop += 1
                    continue
                aimg = res["image"]
                bx0, by0, bx1, by1 = [float(v) for v in res["bboxes"][0]]
                abox = [bx0, by0, bx1 - bx0, by1 - by0]
                if rng.random() < args.occlusion_p:
                    aimg = add_edge_occlusion(aimg, abox, rng)
                name = f"{base}__aug{k}.jpg"
                rel = os.path.join("aug", name)
                cv2.imwrite(os.path.join(BOXES_DIR, rel), aimg,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                ah, aw = aimg.shape[:2]
                aug_records.append({
                    "image": rel.replace("\\", "/"), "w": aw, "h": ah,
                    "label": 1, "split": "train", "group": r["group"],
                    "source": r["source"], "method": "aug",
                    "box": [int(round(v)) for v in abox], "augmented": True,
                    "parent": r["image"],
                })
                n_pos_out += 1

        elif r["label"] == 0:
            def _emit_neg(aimg, suffix, method):
                nonlocal n_neg_out
                name = f"{base}__{suffix}.jpg"
                rel = os.path.join("aug", name)
                cv2.imwrite(os.path.join(BOXES_DIR, rel), aimg,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                ah, aw = aimg.shape[:2]
                aug_records.append({
                    "image": rel.replace("\\", "/"), "w": aw, "h": ah,
                    "label": 0, "split": "train", "group": r["group"],
                    "source": r["source"], "method": method,
                    "box": None, "augmented": True, "parent": r["image"],
                })
                n_neg_out += 1

            # N general augmented copies (rebalance the class ratio)
            for k in range(args.neg_per_image):
                _emit_neg(neg_tf(image=img)["image"], f"negaug{k}", "negaug")
            # plus glare-hardened copies on a fraction (glint != camera)
            if args.neg_glare_frac > 0 and rng.random() < args.neg_glare_frac:
                _emit_neg(add_glare(img, rng), "glare", "glare")

    json.dump({"target_width": manifest.get("target_width"), "images": aug_records},
              open(AUG_MANIFEST, "w"), indent=1)
    print(f"wrote {AUG_MANIFEST}")
    print(f"train positives in : {n_pos_in}")
    print(f"positive variants  : {n_pos_out}  (dropped {n_drop} where box fell out)")
    print(f"negative variants  : {n_neg_out}  (negaug x{args.neg_per_image} + glare)")
    print(f"total augmented    : {len(aug_records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
