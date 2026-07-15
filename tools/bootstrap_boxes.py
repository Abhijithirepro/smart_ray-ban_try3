#!/usr/bin/env python3
"""Auto-bootstrap whole-glasses bounding boxes for object-detection training.

We have no hand-drawn boxes, only folder labels. Rather than annotate the images
by hand, we reuse the repo's own localizer: pipeline.region.locate_region already
returns ONE bbox around the whole glasses (frame + end pieces, camera inside) from
a Haar face anchor. We run every labeled image through preprocess -> locate_region,
save the image, and record the box.

Two detector classes are emitted (record field "cls"):
  * "rayban_meta" — Ray-Ban Meta positives (label 1)
  * "glasses"     — normal eyewear (label 0 with glasses visibly present:
                    normal_glassess, normal_frame, MeGlass, hirepro glasses folders)
  * null          — true NO-GLASSES background (hirepro ".../without glasses" and
                    the data/noglasses/** drop-in dir): saved with box=null so the
                    detector learns "nothing to detect" only on genuinely bare faces.

Output is a single source-of-truth manifest, data/boxes/boxes.json:

    {
      "target_width": 1000,
      "images": [
        {"image": "images/train/pos/ray_ban_face__images (10).jpg",
         "w": 1000, "h": 1779, "label": 1, "cls": "rayban_meta", "split": "train",
         "group": "ray_ban_face:images (10)", "source": "ray_ban_face",
         "method": "worn", "box": [x, y, w, h]},
        {"image": "images/train/neg/normal_glassess__foo.jpg",
         "w": 1000, "h": 1333, "label": 0, "cls": "glasses", "split": "train",
         "group": "...", "source": "normal_glassess", "method": "worn",
         "box": [x, y, w, h]},
        {"image": "images/train/neg/hirepro_images_normal_8-july__without_glasses_10.jpg",
         "w": 1280, "h": 960, "label": 0, "cls": null, "split": "train",
         "group": "...", "source": "hirepro_images_normal/8-july", "box": null},
        ...
      ]
    }

Boxes are pixel [x, y, w, h] in the saved image's coordinates, so no back-mapping
is ever needed — the box always matches the image we ship to the trainer. Boxed
classes that the localizer can't box (tiny/failed — and for the "glasses" class the
untrustworthy "center" fallback too) are DROPPED with a warning rather than kept as
background, so a visible pair of glasses is never trained as "no glasses". Run the
review UI (tools/label_review) afterwards to fix mislocated boxes.

Human-reviewed Ray-Ban boxes in an existing boxes.json are preserved by default
(--merge-existing); pass --no-merge-existing to re-localize everything.

    python tools/bootstrap_boxes.py                 # all sources
    python tools/bootstrap_boxes.py --max-meglass 200 --overlays --noglasses-val-frac 0.4

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
    physical glasses / person lands in the same split (no leakage). A larger
    val_frac buckets a SUPERSET of a smaller one, so raising the fraction keeps
    every previously-val group in val."""
    # stable, seed-free hash (avoids PYTHONHASHSEED variance)
    h = 0
    for ch in group:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    bucket = h % 100
    return "val" if bucket < int(val_frac * 100) else "train"


def _det_class(source: str, label: int, rel_path: str):
    """Detector class for one image: "rayban_meta", "glasses", or None (true
    no-glasses background). The hirepro 8-july folder mixes both negative kinds —
    its "without glasses" sub-folder is the only bare-face source besides the
    data/noglasses drop-in dir."""
    if label == 1:
        return "rayban_meta"
    norm = rel_path.replace("\\", "/")
    if source == "hirepro_images_normal/8-july" and norm.startswith("without glasses"):
        return None
    if source.startswith("data/noglasses"):
        return None
    return "glasses"


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
    ap.add_argument("--noglasses-val-frac", type=float, default=0.4,
                    help="val fraction for true NO-GLASSES background images (default "
                         "0.4 — the set is small, hold more out for honest eval)")
    ap.add_argument("--merge-existing", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="keep boxes from the current boxes.json for Ray-Ban records "
                         "(preserves label_review corrections) instead of re-running "
                         "the localizer on them (default on)")
    args = ap.parse_args(argv)

    cfg = Config()
    os.makedirs(IMG_DIR, exist_ok=True)
    if args.overlays:
        os.makedirs(OVERLAY_DIR, exist_ok=True)

    # Previously-saved (possibly human-reviewed) Ray-Ban boxes, keyed by the saved
    # file's basename (stable across split changes).
    prev_boxes = {}
    prev_manifest = os.path.join(OUT_DIR, "boxes.json")
    if args.merge_existing and os.path.isfile(prev_manifest):
        for r in json.load(open(prev_manifest))["images"]:
            if r.get("label") == 1 and r.get("box"):
                prev_boxes[os.path.basename(r["image"])] = r

    records = []
    stats = {"boxed": {"rayban_meta": 0, "glasses": 0}, "background": 0,
             "dropped": {"rayban_meta": 0, "glasses": 0}, "merged": 0,
             "meglass_skipped": 0, "read_error": 0, "methods": {}}
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

            rel_src = os.path.relpath(path, abs_folder)
            det_cls = _det_class(source, label, rel_src)

            frac = args.noglasses_val_frac if det_cls is None else args.val_frac
            split = _split_for_group(group, frac)
            tag = "pos" if label == 1 else "neg"
            name = _safe_name(source, rel_src) + ".jpg"
            rel = os.path.join("images", split, tag, name)
            dst = os.path.join(OUT_DIR, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            box = None
            method = None
            if det_cls is not None:
                prev = prev_boxes.get(name)
                if (prev is not None and det_cls == "rayban_meta"
                        and prev.get("w") == W and prev.get("h") == H):
                    # keep the stored (possibly label_review-corrected) box
                    box = list(prev["box"])
                    method = prev.get("method") or "reviewed"
                    stats["merged"] += 1
                else:
                    reg = region.locate_region(fr, cfg)
                    method = reg.method
                    if reg.w < 20 or reg.h < 20 or (det_cls == "glasses"
                                                    and method == "center"):
                        # never keep a visible pair of glasses as background
                        print(f"  drop (no {det_cls} box, method={method}): {path}",
                              file=sys.stderr)
                        stats["dropped"][det_cls] += 1
                        continue
                    # scale the canonical box into the saved (original) image's coords
                    bx = max(0, min(W, reg.x * can_to_save))
                    by = max(0, min(H, reg.y * can_to_save))
                    bw = min(W - bx, reg.w * can_to_save)
                    bh = min(H - by, reg.h * can_to_save)
                    box = [int(round(bx)), int(round(by)),
                           int(round(bw)), int(round(bh))]
                stats["methods"][method] = stats["methods"].get(method, 0) + 1
                stats["boxed"][det_cls] += 1
            else:
                stats["background"] += 1

            cv2.imwrite(dst, save_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

            if args.overlays and box is not None:
                ov = save_img.copy()
                x, y, w, h = box
                color = (0, 255, 0) if det_cls == "rayban_meta" else (0, 165, 255)
                cv2.rectangle(ov, (x, y), (x + w, y + h), color, 3)
                cv2.putText(ov, f"{det_cls} [{method}]", (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imwrite(os.path.join(OVERLAY_DIR, name), ov)

            records.append({
                "image": rel.replace("\\", "/"),
                "w": W, "h": H,
                "label": label,
                "cls": det_cls,
                "split": split,
                "group": group,
                "source": source,
                "method": method,
                "box": box,
            })

    manifest = {"target_width": cfg.target_width, "images": records}
    with open(os.path.join(OUT_DIR, "boxes.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    # Split summary (per detector class)
    def _n(cls, split):
        return sum(1 for r in records if r["cls"] == cls and r["split"] == split)

    print(f"\nwrote {os.path.join(OUT_DIR, 'boxes.json')}")
    print(f"rayban_meta boxed: {stats['boxed']['rayban_meta']}  "
          f"(dropped {stats['dropped']['rayban_meta']}, "
          f"kept from review {stats['merged']})")
    print(f"glasses boxed    : {stats['boxed']['glasses']}  "
          f"(dropped {stats['dropped']['glasses']}, "
          f"meglass skipped {stats['meglass_skipped']})")
    print(f"no-glasses bg    : {stats['background']}")
    print(f"read errors      : {stats['read_error']}")
    print(f"localize methods : {stats['methods']}")
    print(f"split  train: rayban={_n('rayban_meta', 'train')} "
          f"glasses={_n('glasses', 'train')} noglasses={_n(None, 'train')}   "
          f"val: rayban={_n('rayban_meta', 'val')} "
          f"glasses={_n('glasses', 'val')} noglasses={_n(None, 'val')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
