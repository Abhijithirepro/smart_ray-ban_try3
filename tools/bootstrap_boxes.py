#!/usr/bin/env python3
"""Auto-bootstrap whole-glasses bounding boxes for object-detection training.

We have no hand-drawn boxes, only folder labels. Rather than annotate ~80 Ray-Ban
images by hand, we reuse the repo's own localizer: pipeline.region.locate_region
already returns ONE bbox around the whole glasses (frame + end pieces, camera
inside) from a Haar face anchor. We run every labeled image through
preprocess -> locate_region, save the canonical (target_width-wide) color image,
and record the box.

Output is a single source-of-truth manifest, data/boxes/boxes.json:

    {
      "target_width": 1000,
      "images": [
        {"image": "images/train/pos/ray_ban_face__images (10).jpg",
         "w": 1000, "h": 1779, "label": 1, "split": "train",
         "group": "ray_ban_face:images (10)", "source": "ray_ban_face",
         "method": "worn", "box": [x, y, w, h]},        # positives only
        {"image": "images/train/neg/normal_glassess__foo.jpg",
         "w": 1000, "h": 1333, "label": 0, "split": "train",
         "group": "...", "source": "normal_glassess", "box": null},
        ...
      ]
    }

Boxes are pixel [x, y, w, h] in the saved (canonical) image's coordinates, so no
back-mapping is ever needed — the box always matches the image we ship to the
trainer. Positives that the localizer can't box (tiny/failed) are dropped with a
warning; run the review UI (tools/label_review) afterwards to fix mislocated boxes.

    python tools/bootstrap_boxes.py                 # all sources
    python tools/bootstrap_boxes.py --max-meglass 200 --overlays

The optional --overlays flag dumps annotated previews to data/boxes/overlays/ for a
quick sanity check.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

# Make the repo root importable when run as `python tools/bootstrap_boxes.py`.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from config import Config                                    # noqa: E402
from datasets import SOURCES, _files, _group_key, _source_of  # noqa: E402
from pipeline import preprocess, region                      # noqa: E402

OUT_DIR = os.path.join(_REPO, "data", "boxes")
IMG_DIR = os.path.join(OUT_DIR, "images")
OVERLAY_DIR = os.path.join(OUT_DIR, "overlays")


def _safe_name(source: str, rel_path: str) -> str:
    """Flatten a source folder + source-relative path into a unique, filesystem-safe
    stem. The full sub-path (not just the basename) is folded in so duplicate
    basenames across sub-folders — e.g. hirepro_images_normal/8-july/<lighting>/image
    (5).png appearing under three lighting dirs — produce distinct names instead of
    silently overwriting one another."""
    stem = os.path.splitext(rel_path)[0]
    src = source.replace("/", "_").replace("\\", "_")
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)
    return f"{src}__{safe}"


def _split_for_group(group: str, val_frac: float) -> str:
    """Deterministic group-aware split: hash the group so every image of one
    physical glasses / person lands in the same split (no leakage)."""
    # stable, seed-free hash (avoids PYTHONHASHSEED variance)
    h = 0
    for ch in group:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    bucket = h % 100
    return "val" if bucket < int(val_frac * 100) else "train"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction of GROUPS held out for validation (default 0.2)")
    ap.add_argument("--max-meglass", type=int, default=200,
                    help="cap MeGlass bulk negatives so they don't swamp training "
                         "(default 200; use 0 for all 500)")
    ap.add_argument("--overlays", action="store_true",
                    help="also write annotated preview images to data/boxes/overlays/")
    ap.add_argument("--max-width", type=int, default=0,
                    help="optional safety cap on saved image width (0 = keep ORIGINAL "
                         "resolution, the default). Localization always runs on an "
                         "internal canonical copy; the SAVED image + box are original.")
    args = ap.parse_args(argv)

    cfg = Config()
    os.makedirs(IMG_DIR, exist_ok=True)
    if args.overlays:
        os.makedirs(OVERLAY_DIR, exist_ok=True)

    records = []
    stats = {"pos": 0, "neg": 0, "pos_dropped": 0, "meglass_skipped": 0,
             "read_error": 0, "methods": {}}
    meglass_kept = 0

    for folder, label, dist in SOURCES:
        abs_folder = os.path.join(_REPO, folder)
        if not os.path.isdir(abs_folder):
            continue
        for path in _files(abs_folder):
            source = _source_of(folder, path)
            group = _group_key(source, path)

            # Cap MeGlass bulk negatives (there are 500 tiny 120px faces).
            if args.max_meglass and source == "data/normal/meglass":
                if meglass_kept >= args.max_meglass:
                    stats["meglass_skipped"] += 1
                    continue
                meglass_kept += 1

            try:
                # Localization runs on the canonical (resized) copy — Haar needs a
                # consistent scale — but we SAVE the image at its ORIGINAL resolution
                # so the detector trains on full-detail images, not a 1000px downscale.
                fr = preprocess.preprocess(path, cfg)
                orig = preprocess._flatten_alpha(preprocess.load_image(path))
            except Exception as e:  # noqa: BLE001
                print(f"  skip (read error): {path} :: {e}", file=sys.stderr)
                stats["read_error"] += 1
                continue

            # optional safety cap; default keeps original resolution
            save_img = orig
            out_scale = 1.0                       # canonical px -> saved px
            if args.max_width and orig.shape[1] > args.max_width:
                out_scale = args.max_width / float(orig.shape[1])
                nh = max(1, round(orig.shape[0] * out_scale))
                save_img = cv2.resize(orig, (args.max_width, nh),
                                      interpolation=cv2.INTER_AREA)
            H, W = save_img.shape[:2]
            # map a canonical-space value to the saved image's pixels
            can_to_save = out_scale / fr.scale    # (1/fr.scale)=canonical->orig, then *out_scale

            split = _split_for_group(group, args.val_frac)
            tag = "pos" if label == 1 else "neg"
            name = _safe_name(source, os.path.relpath(path, abs_folder)) + ".jpg"
            rel = os.path.join("images", split, tag, name)
            dst = os.path.join(OUT_DIR, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            box = None
            method = None
            if label == 1:
                reg = region.locate_region(fr, cfg)
                method = reg.method
                stats["methods"][method] = stats["methods"].get(method, 0) + 1
                if reg.w < 20 or reg.h < 20:
                    print(f"  drop (no box): {path}", file=sys.stderr)
                    stats["pos_dropped"] += 1
                    continue
                # scale the canonical box into the saved (original) image's coords
                bx = max(0, min(W, reg.x * can_to_save))
                by = max(0, min(H, reg.y * can_to_save))
                bw = min(W - bx, reg.w * can_to_save)
                bh = min(H - by, reg.h * can_to_save)
                box = [int(round(bx)), int(round(by)), int(round(bw)), int(round(bh))]
                stats["pos"] += 1
            else:
                stats["neg"] += 1

            cv2.imwrite(dst, save_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

            if args.overlays and box is not None:
                ov = save_img.copy()
                x, y, w, h = box
                cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(ov, f"rayban_meta [{method}]", (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(OVERLAY_DIR, name), ov)

            records.append({
                "image": rel.replace("\\", "/"),
                "w": W, "h": H,
                "label": label,
                "split": split,
                "group": group,
                "source": source,
                "method": method,
                "box": box,
            })

    manifest = {"target_width": cfg.target_width, "images": records}
    with open(os.path.join(OUT_DIR, "boxes.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    # Split summary
    tr_pos = sum(1 for r in records if r["label"] == 1 and r["split"] == "train")
    va_pos = sum(1 for r in records if r["label"] == 1 and r["split"] == "val")
    tr_neg = sum(1 for r in records if r["label"] == 0 and r["split"] == "train")
    va_neg = sum(1 for r in records if r["label"] == 0 and r["split"] == "val")
    print(f"\nwrote {os.path.join(OUT_DIR, 'boxes.json')}")
    print(f"positives boxed : {stats['pos']}  (dropped {stats['pos_dropped']})")
    print(f"negatives       : {stats['neg']}  (meglass skipped {stats['meglass_skipped']})")
    print(f"read errors     : {stats['read_error']}")
    print(f"localize methods: {stats['methods']}")
    print(f"split  train: pos={tr_pos} neg={tr_neg}   val: pos={va_pos} neg={va_neg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
