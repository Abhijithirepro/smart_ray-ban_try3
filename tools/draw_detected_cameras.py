#!/usr/bin/env python3
"""Batch-draw the PRECISE camera-module location the corner CNN found, per image.

Runs the shipped detector (models/corner_clf.pt over the face-anchored top-outer
corner grid, pipeline/corners.py). For each side that fires (P(module) >= tc_maybe)
we find the exact pixels that drove the "module" decision in two steps:

  1. OCCLUSION SENSITIVITY over the best candidate crop - slide a gray patch and
     watch how much the module logit drops; the peak drop is the pixel the model
     actually relies on. This is model-faithful and far sharper than Grad-CAM over
     the last conv block (only ~4x4 cells on a 112px crop, which gave a loose,
     mis-placed blob - see git history).
  2. SNAP to the physical module - around that peak, in the ORIGINAL full-res
     image, look for the small dark, glossy, near-circular blob (HoughCircles +
     darkness/glint scoring). The circle bbox is the TIGHT box we draw; if none is
     found we fall back to a small box on the occlusion peak.

The faint search region is drawn for context. Header shows the three-tier verdict.

    python tools/draw_detected_cameras.py     # (run with PYTHONPATH=. from repo root)

Outputs -> detected_image_draw/{normal,rayban}/<name>.png
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

from config import Config
from pipeline import corners

SRC = {"normal": "normal_glassess", "rayban": "ray_ban_face"}
OUT_ROOT = "detected_image_draw"
MODEL_PATH = "models/corner_clf.pt"
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 210, 255)
FAINT = (120, 120, 120)
CYAN = (255, 255, 0)

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# uint8 gray fill used to occlude a patch (ImageNet mean colour = "no information")
OCC_FILL = np.array([124, 116, 104], np.uint8)


def build_model():
    m = tv.models.mobilenet_v3_small(weights=None)
    in_f = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_f, 1)
    return m


def load_detector():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    m = build_model()
    m.load_state_dict(ckpt["state_dict"])
    m.to(DEVICE).eval()
    return m, ckpt


def to_tensor(rgb, n):
    """RGB uint8 HxWx3 -> normalized 1x3xn x n float tensor on DEVICE."""
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t = F.interpolate(t, size=(n, n), mode="bilinear", align_corners=False)
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t.to(DEVICE)


@torch.no_grad()
def score_batch(model, rgb_crops, n):
    if not rgb_crops:
        return np.zeros(0, np.float32)
    batch = torch.cat([to_tensor(c, n) for c in rgb_crops], 0)
    return torch.sigmoid(model(batch)).cpu().numpy().ravel()


@torch.no_grad()
def logit_batch(model, rgb_crops, n):
    """Raw module logits (not sigmoid) - occlusion differences live in logit space."""
    batch = torch.cat([to_tensor(c, n) for c in rgb_crops], 0)
    return model(batch).cpu().numpy().ravel()


def occlusion_map(model, rgb_crop, n):
    """Occlusion sensitivity heatmap (n x n, [0,1], canonical/crop orientation).

    Slide a gray patch over the crop; heat = drop in the module logit when that
    region is hidden. High heat = pixels the 'module' decision depends on. Batched
    in a single forward pass. Sharper and more faithful than last-block Grad-CAM."""
    crop = cv2.resize(rgb_crop, (n, n), interpolation=cv2.INTER_CUBIC)
    base = float(logit_batch(model, [crop], n)[0])

    p = max(8, n // 6)                 # patch side (~18px at n=112)
    stride = max(4, p // 3)            # ~6px
    xs = list(range(0, n - p + 1, stride)) or [0]
    ys = list(range(0, n - p + 1, stride)) or [0]
    if xs[-1] != n - p:
        xs.append(n - p)
    if ys[-1] != n - p:
        ys.append(n - p)

    variants, spans = [], []
    for y in ys:
        for x in xs:
            occ = crop.copy()
            occ[y:y + p, x:x + p] = OCC_FILL
            variants.append(occ)
            spans.append((x, y))
    logits = logit_batch(model, variants, n)

    heat = np.zeros((n, n), np.float32)
    cnt = np.zeros((n, n), np.float32)
    for (x, y), lg in zip(spans, logits):
        heat[y:y + p, x:x + p] += (base - lg)   # bigger drop -> more important
        cnt[y:y + p, x:x + p] += 1.0
    heat = heat / np.maximum(cnt, 1e-6)
    heat = np.maximum(heat, 0.0)                 # only spots that HELP the module call
    lo, hi = float(heat.min()), float(heat.max())
    return (heat - lo) / (hi - lo) if hi > lo else np.zeros((n, n), np.float32)


def peak_canonical(heat):
    """(px, py) of the occlusion peak, in canonical crop pixels."""
    py, px = np.unravel_index(int(heat.argmax()), heat.shape)
    return int(px), int(py)


def map_point(canon_pt, box, side, n):
    """Map a canonical-crop point (px,py in n-space) to original-image px. The R
    side crop was horizontally flipped to canonical orientation, so un-flip x."""
    bx, by, bw, bh = box
    fx = canon_pt[0] / n
    if side == "R":
        fx = 1.0 - fx
    fy = canon_pt[1] / n
    return int(round(bx + fx * bw)), int(round(by + fy * bh))


def snap_to_module(image_bgr, peak_xy, face, n):
    """Snap the occlusion peak to the physical camera module in the ORIGINAL image.

    The module is a small dark, glossy, near-circular blob. Search a face-relative
    neighbourhood around the peak with HoughCircles and score candidates by interior
    darkness * proximity-to-peak * glint. Returns (box=(x0,y0,x1,y1), center=(cx,cy)).
    Falls back to a small box on the peak if no good circle is found - still far
    tighter and better-placed than the old Grad-CAM blob."""
    H, W = image_bgr.shape[:2]
    px, py = peak_xy
    reach = max(12, int(round(0.15 * face.w)))          # neighbourhood half-size
    x0 = max(0, px - reach); y0 = max(0, py - reach)
    x1 = min(W, px + reach); y1 = min(H, py + reach)
    patch = image_bgr[y0:y1, x0:x1]

    fallback_r = max(6, int(round(0.06 * face.w)))
    fallback = ((px - fallback_r, py - fallback_r, px + fallback_r, py + fallback_r),
                (px, py))
    if patch.size == 0:
        return fallback

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.medianBlur(gray, 3)
    min_r = max(3, int(round(0.020 * face.w)))
    max_r = max(min_r + 2, int(round(0.075 * face.w)))
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=max(8, min_r),
                               param1=100, param2=18,
                               minRadius=min_r, maxRadius=max_r)
    if circles is None:
        return fallback

    best, best_score = None, -1.0
    for cx, cy, r in np.round(circles[0]).astype(int):
        r = int(r)
        # interior stats via a circular mask
        mask = np.zeros(gray.shape, np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        vals = gray[mask == 255]
        if vals.size == 0:
            continue
        darkness = 1.0 - float(vals.mean()) / 255.0        # dark module -> high
        glint = float(vals.max()) / 255.0                  # specular highlight -> high
        gcx, gcy = x0 + cx, y0 + cy                        # back to original px
        dist = float(np.hypot(gcx - px, gcy - py))
        proximity = float(np.exp(-dist / max(1.0, 0.12 * face.w)))
        score = darkness * (0.55 + 0.45 * glint) * proximity
        if score > best_score:
            best_score, best = score, (gcx, gcy, r)

    if best is None or best_score <= 0.0:
        return fallback
    gcx, gcy, r = best
    pad = int(round(0.20 * r))
    return ((gcx - r - pad, gcy - r - pad, gcx + r + pad, gcy + r + pad), (gcx, gcy))


def tier_verdict(probs, tc_hi, tc_maybe):
    if sum(1 for p in probs if p >= tc_hi) >= 2:
        return "YES"
    if any(p >= tc_maybe for p in probs):
        return "MAYBE"
    return "NO"


def banner(img, text, color):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2,
                cv2.LINE_AA)


def process(path, model, cfg, ckpt):
    img = cv2.imread(path)
    if img is None:
        return None
    n = ckpt["input"]
    tc_hi, tc_maybe = ckpt["tc_hi"], ckpt["tc_maybe"]

    corn = corners.extract(img, cfg)
    if corn is None:
        banner(img, "NO FACE DETECTED", FAINT)
        return img, "NOFACE"

    side_score = {}
    for side in ("L", "R"):
        crops, boxes = corn.grid[side], corn.grid_boxes[side]
        probs = score_batch(model, crops, n)
        if len(probs) == 0:
            side_score[side] = 0.0
            continue
        bi = int(np.argmax(probs))
        p = float(probs[bi]); side_score[side] = p
        region = boxes[bi]
        # faint search region for context
        rx, ry, rw, rh = region
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), FAINT, 1)
        if p < tc_maybe:                       # no module here -> nothing to pinpoint
            continue
        heat = occlusion_map(model, crops[bi], n)
        peak_orig = map_point(peak_canonical(heat), region, side, n)
        obox, opt = snap_to_module(img, peak_orig, corn.face, n)
        col = GREEN if p >= tc_hi else YELLOW
        cv2.rectangle(img, (obox[0], obox[1]), (obox[2], obox[3]), col, 2)
        cv2.drawMarker(img, opt, CYAN, cv2.MARKER_CROSS, 14, 2)
        cv2.putText(img, f"camera P={p:.2f}", (obox[0], max(12, obox[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)

    verdict = tier_verdict([side_score["L"], side_score["R"]], tc_hi, tc_maybe)
    vcol = {"YES": GREEN, "MAYBE": YELLOW, "NO": RED}[verdict]
    banner(img, f"{verdict}   P(module) L={side_score['L']:.2f} "
                f"R={side_score['R']:.2f}", vcol)
    return img, verdict


def main():
    cfg = Config()
    model, ckpt = load_detector()
    print(f"device={DEVICE}  tc_hi={ckpt['tc_hi']:.3f}  tc_maybe={ckpt['tc_maybe']:.3f}")

    for label, src in SRC.items():
        out_dir = os.path.join(OUT_ROOT, label)
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(fn for fn in os.listdir(src) if fn.lower().endswith(EXTS))
        counts = {"YES": 0, "MAYBE": 0, "NO": 0, "NOFACE": 0}
        for fn in files:
            res = process(os.path.join(src, fn), model, cfg, ckpt)
            if res is None:
                print(f"  skip (unreadable): {src}/{fn}")
                continue
            out, verdict = res
            counts[verdict] += 1
            stem = os.path.splitext(fn)[0]
            cv2.imwrite(os.path.join(out_dir, f"{stem}.png"), out)
        print(f"{label}: wrote {len(files)} -> {out_dir}/  "
              f"YES={counts['YES']} MAYBE={counts['MAYBE']} "
              f"NO={counts['NO']} NOFACE={counts['NOFACE']}")


if __name__ == "__main__":
    main()
