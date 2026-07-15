#!/usr/bin/env python3
"""Render the box manifests into a trainable detection dataset.

Merges the reviewed real manifest (data/boxes/boxes.json) with the offline
augmentations (data/boxes/aug.json) and writes one of two on-disk layouts:

  --format coco   (for RF-DETR)   data/rayban_coco/{train,valid,test}/
                                    each: images + _annotations.coco.json
  --format yolo   (for YOLO12)    data/rayban_yolo/
                                    images/{train,val}/  labels/{train,val}/  rayban.yaml
  --format yolox  (for YOLOX)     data/rayban_yolox_coco/
                                    {train2017,val2017}/  +
                                    annotations/instances_{train2017,val2017}.json

Augmentations only exist for the train split, so validation stays 100% real.
No-glasses images are written as background: COCO images with zero annotations /
empty YOLO .txt files. Two classes: rayban_meta (id 1) + glasses (id 2); records
without a "cls" field (old manifests) default to rayban_meta.

    python tools/export_dataset.py --format coco
    python tools/export_dataset.py --format yolo
    python tools/export_dataset.py --format yolox
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOXES_DIR = os.path.join(_REPO, "data", "boxes")
CATEGORIES = [{"id": 1, "name": "rayban_meta", "supercategory": "none"},
              {"id": 2, "name": "glasses", "supercategory": "none"}]
CAT_ID = {c["name"]: c["id"] for c in CATEGORIES}


def _cat_id(rec):
    """COCO category id for a boxed record; old manifests lack "cls" (their boxed
    records were all Ray-Ban positives)."""
    return CAT_ID[rec.get("cls") or "rayban_meta"]


def load_records():
    real = json.load(open(os.path.join(BOXES_DIR, "boxes.json")))["images"]
    aug_path = os.path.join(BOXES_DIR, "aug.json")
    aug = json.load(open(aug_path))["images"] if os.path.isfile(aug_path) else []
    return real + aug


def _flat_name(rel):
    return rel.replace("/", "__").replace("\\", "__")


def export_coco(records, out_dir):
    # RF-DETR expects train/valid/test; we mirror val -> test.
    split_map = {"train": "train", "val": "valid"}
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    coco = {s: {"images": [], "annotations": [], "categories": CATEGORIES}
            for s in ("train", "valid", "test")}
    img_id = {s: 0 for s in coco}
    ann_id = {s: 0 for s in coco}

    def add(split, rec):
        nonlocal img_id, ann_id
        src = os.path.join(BOXES_DIR, rec["image"])
        if not os.path.isfile(src):
            return
        name = _flat_name(rec["image"])
        dst_dir = os.path.join(out_dir, split)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, os.path.join(dst_dir, name))
        img_id[split] += 1
        iid = img_id[split]
        coco[split]["images"].append(
            {"id": iid, "file_name": name, "width": rec["w"], "height": rec["h"]})
        if rec.get("box"):
            x, y, w, h = rec["box"]
            ann_id[split] += 1
            coco[split]["annotations"].append(
                {"id": ann_id[split], "image_id": iid, "category_id": _cat_id(rec),
                 "bbox": [x, y, w, h], "area": int(w * h), "iscrowd": 0,
                 "segmentation": []})

    for rec in records:
        split = split_map.get(rec["split"])
        if split is None:
            continue
        add(split, rec)
        if split == "valid":     # test mirrors valid (RF-DETR wants a test split)
            add("test", rec)

    for s in coco:
        with open(os.path.join(out_dir, s, "_annotations.coco.json"), "w") as fh:
            json.dump(coco[s], fh)
        print(f"  {s:5s}: {len(coco[s]['images'])} images, "
              f"{len(coco[s]['annotations'])} boxes")
    print(f"wrote COCO dataset -> {out_dir}")


def export_yolox(records, out_dir):
    """YOLOX COCODataset layout: images under data_dir/{train2017,val2017}/ and
    annotations under data_dir/annotations/instances_{train2017,val2017}.json.
    No-glasses images are kept as zero-annotation images (YOLOX trains them as
    background)."""
    split_map = {"train": "train2017", "val": "val2017"}
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    ann_dir = os.path.join(out_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    coco = {s: {"images": [], "annotations": [], "categories": CATEGORIES}
            for s in ("train2017", "val2017")}
    img_id = {s: 0 for s in coco}
    ann_id = {s: 0 for s in coco}

    for rec in records:
        split = split_map.get(rec["split"])
        if split is None:
            continue
        src = os.path.join(BOXES_DIR, rec["image"])
        if not os.path.isfile(src):
            continue
        name = _flat_name(rec["image"])
        dst_dir = os.path.join(out_dir, split)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, os.path.join(dst_dir, name))
        img_id[split] += 1
        iid = img_id[split]
        coco[split]["images"].append(
            {"id": iid, "file_name": name, "width": rec["w"], "height": rec["h"]})
        if rec.get("box"):
            x, y, w, h = rec["box"]
            ann_id[split] += 1
            coco[split]["annotations"].append(
                {"id": ann_id[split], "image_id": iid, "category_id": _cat_id(rec),
                 "bbox": [x, y, w, h], "area": int(w * h), "iscrowd": 0,
                 "segmentation": []})

    for s in coco:
        with open(os.path.join(ann_dir, f"instances_{s}.json"), "w") as fh:
            json.dump(coco[s], fh)
        print(f"  {s:9s}: {len(coco[s]['images'])} images, "
              f"{len(coco[s]['annotations'])} boxes")
    print(f"wrote YOLOX COCO dataset -> {out_dir}")


def export_yolo(records, out_dir):
    split_map = {"train": "train", "val": "val"}
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = os.path.join(out_dir, sub)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    counts = {"train": [0, 0], "val": [0, 0]}   # [images, boxes]

    for rec in records:
        split = split_map.get(rec["split"])
        if split is None:
            continue
        src = os.path.join(BOXES_DIR, rec["image"])
        if not os.path.isfile(src):
            continue
        name = _flat_name(rec["image"])
        stem = os.path.splitext(name)[0]
        shutil.copy(src, os.path.join(out_dir, "images", split, name))
        lines = []
        if rec.get("box"):
            x, y, w, h = rec["box"]
            W, H = rec["w"], rec["h"]
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            nw, nh = w / W, h / H
            lines.append(f"{_cat_id(rec) - 1} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            counts[split][1] += 1
        with open(os.path.join(out_dir, "labels", split, stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines))
        counts[split][0] += 1

    names = "\n".join(f"  {c['id'] - 1}: {c['name']}" for c in CATEGORIES)
    yaml = (f"path: {out_dir}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names:\n{names}\n")
    with open(os.path.join(out_dir, "rayban.yaml"), "w") as fh:
        fh.write(yaml)
    for s in ("train", "val"):
        print(f"  {s:5s}: {counts[s][0]} images, {counts[s][1]} boxes")
    print(f"wrote YOLO dataset -> {out_dir}  (rayban.yaml)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", choices=["coco", "yolo", "yolox"], required=True)
    ap.add_argument("--out", default=None, help="output dir (default per format)")
    args = ap.parse_args(argv)

    records = load_records()
    if args.format == "coco":
        out = args.out or os.path.join(_REPO, "data", "rayban_coco")
        export_coco(records, out)
    elif args.format == "yolox":
        out = args.out or os.path.join(_REPO, "data", "rayban_yolox_coco")
        export_yolox(records, out)
    else:
        out = args.out or os.path.join(_REPO, "data", "rayban_yolo")
        export_yolo(records, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
