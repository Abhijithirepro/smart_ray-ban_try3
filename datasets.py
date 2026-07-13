"""Dataset index + region-crop loader for the CNN glasses classifier.

Labels come from folder names (rayban=1, normal=0). Both the original 5 folders
and any drop-in `data/rayban/**` `data/normal/**` are included, so new data just
needs to be dropped into those dirs. Each image is localized to its glasses region
(pipeline/region) and returned as an RGB crop; grouping avoids near-duplicate
product angles straddling a CV split.

    from datasets import load_dataset
    items = load_dataset(Config())   # list of Item(rgb, label, group, dist, path)
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import cv2
import numpy as np

from config import Config
from pipeline import preprocess, region

# (folder, label, distribution). label: 1=rayban, 0=normal.
SOURCES = [
    ("ray_ban_frame", 1, "studio"),
    ("ray_ban_face", 1, "real"),
    ("actual_rayban_cases", 1, "real"),
    ("normal_frame", 0, "studio"),
    ("normal_glassess", 0, "real"),
    ("hirepro_images_normal/8-july", 0, "real"),  # HirePro normal/no-glasses negatives
    ("data/rayban", 1, "real"),      # drop-in: new Ray-Ban images
    ("data/normal", 0, "real"),      # drop-in: new normal images
]
EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclass
class Item:
    rgb: np.ndarray      # HxWx3 uint8 region crop (RGB)
    label: int
    group: str           # CV grouping key (keeps same physical glasses together)
    dist: str            # "studio" | "real"
    path: str
    source: str          # originating SOURCES folder (e.g. "normal_glassess",
                         # "data/normal/meglass") — drives the two-tier FP threshold


def meglass_identity(stem):
    """Identity key of a MeGlass filename: the string before the SECOND '@' (the
    MegaFace convention), e.g. '10032527@N08_identity_4@2897031059_1' -> '10032527@N08_identity_4'.
    Grouping by identity keeps every photo of one person in a single CV fold.
    Returns None if the name doesn't match, so the caller groups per-file.
    Kept in sync with tools/import_meglass.py."""
    parts = stem.split("@")
    return "@".join(parts[:2]) if len(parts) >= 3 else None


def _files(folder):
    out = []
    for ext in EXTS:
        out += glob.glob(os.path.join(folder, "**", "*" + ext), recursive=True)
        out += glob.glob(os.path.join(folder, "*" + ext))
    return sorted(set(out))


def _group_key(folder, path):
    """Keep the same physical glasses (or the same person) in one group. Ray-Ban
    studio shots share a product-code prefix (0rw4010__601_mf__...); MeGlass shots
    share an uploader-identity prefix (10004979@N04_...); everything else groups
    per-file."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if folder == "ray_ban_frame":
        parts = stem.split("__")
        if len(parts) >= 2:
            return "rbf:" + "__".join(parts[:2])
    if folder == "data/normal/meglass":
        ident = meglass_identity(stem)
        if ident:
            return "meglass:" + ident
    return folder + ":" + stem


def _source_of(folder, path):
    """Refine the SOURCES folder into the effective source for an item. MeGlass
    lives under the generic data/normal drop-in dir but is a distinct BULK negative
    tier for the FP-first threshold, so it gets its own source tag."""
    norm = path.replace("\\", "/")
    if folder == "data/normal" and "/meglass/" in norm:
        return "data/normal/meglass"
    return folder


def load_dataset(cfg: Config, verbose=False):
    """Localize + crop every labeled image. Returns list[Item]."""
    items = []
    for folder, label, dist in SOURCES:
        if not os.path.isdir(folder):
            continue
        for path in _files(folder):
            try:
                fr = preprocess.preprocess(path, cfg)
                reg = region.locate_region(fr, cfg)
                crop = region.crop_region(fr, reg)
                if crop.size == 0 or reg.w < 20 or reg.h < 20:
                    continue
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                src = _source_of(folder, path)
                items.append(Item(rgb=rgb, label=label,
                                  group=_group_key(src, path), dist=dist,
                                  path=path, source=src))
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print("skip", path, e)
    return items


if __name__ == "__main__":
    it = load_dataset(Config())
    pos = sum(1 for x in it if x.label == 1)
    real = sum(1 for x in it if x.dist == "real")
    groups = len({x.group for x in it})
    print(f"{len(it)} crops | rayban={pos} normal={len(it)-pos} | "
          f"real={real} studio={len(it)-real} | groups={groups}")
