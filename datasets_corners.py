"""Per-corner dataset for the top-outer camera-module CNN, with a candidate GRID.

A single fixed corner box slides off the module on tilted / oddly-framed heads, so
each side is localized as a small GRID of candidate boxes (pipeline/corners) and
inference takes the max P(module) over the grid. Training must match:

  * POSITIVES contribute only their CENTER box (label 1). Off-center grid crops of
    a Ray-Ban may not contain the module, so they are NOT trained as positives
    (that was the label-noise bug). A side whose module is not visible even at
    grid-max is EXCLUDED (tools/make_corner_exclusions.py).
  * NEGATIVES (in-domain normals) contribute the WHOLE grid as label-0 hard
    negatives — a normal corner has no module at ANY scan position, so this teaches
    the model to reject the module-like normal patches (thick hinges, bright rims)
    that grid scanning would otherwise surface as false positives.
  * MeGlass bulk negatives are 120px low-res, so only their CENTER box is used
    (off-center crops would be degenerate); they are grid-scored at eval for sanity.

Every candidate crop is kept for EVAL (grid-max per image+side); `train_use` marks
which crops are training samples. Face-anchored -> covers the WORN case; no-face
images (studio/held) are skipped.

    from datasets_corners import load_corner_dataset
    items = load_corner_dataset(Config())   # list of CornerItem (one per candidate)
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from dataclasses import dataclass

import cv2
import numpy as np

# Parallelise the load across cores: the per-image cost is cv2 Haar face detection
# (corners.extract). We use a PROCESS pool (not threads) because cv2.CascadeClassifier
# is not safe to call concurrently on one shared instance — each worker process builds
# its own cascades. Output is identical regardless of parallelism: dedup runs serially
# first and ProcessPoolExecutor.map preserves input order, so the CV split and seeded
# sampler see the same item sequence and the trained model is unaffected.
LOAD_WORKERS = int(os.environ.get("LOAD_WORKERS", min(8, (os.cpu_count() or 2))))


def _worker_init():
    cv2.setNumThreads(1)   # each process is one pool slot; don't oversubscribe cores

from config import Config
from datasets import SOURCES, _files, _group_key, _source_of
from pipeline import corners

EXCLUSIONS_PATH = os.path.join("models", "corner_exclusions.json")
MEGLASS_SOURCE = "data/normal/meglass"


@dataclass
class CornerItem:
    rgb: np.ndarray      # HxWx3 uint8 corner crop (RGB), canonical orientation
    label: int
    side: str            # "L" | "R"
    image_id: str        # unique per source image (aggregate corners -> verdict)
    group: str           # CV grouping key (image, or MeGlass identity)
    dist: str            # "studio" | "real"
    source: str
    path: str
    box_idx: int         # index into the side's candidate grid (-1 = center box)
    is_center: bool      # the canonical single box (the training positive)
    excluded: bool = False   # module not visible on this side even at grid-max
    train_use: bool = True   # whether this crop is a TRAINING sample


def _load_exclusions():
    """Set of '<image_id>:<side>' keys whose side is excluded from training as a
    positive (module not visible there — see tools/make_corner_exclusions.py)."""
    if not os.path.exists(EXCLUSIONS_PATH):
        return set()
    return set(json.load(open(EXCLUSIONS_PATH)).get("excluded", []))


def _emit_side(items, c, side, label, image_id, gkey, dist, src, path, excl):
    """Append the center crop + grid crops for one side with train_use flags."""
    key = image_id + ":" + side
    excluded = key in excl
    center_rgb = c.left if side == "L" else c.right
    is_meglass = (src == MEGLASS_SOURCE)

    def add(rgb, box_idx, is_center, train_use):
        items.append(CornerItem(rgb=rgb, label=label, side=side, image_id=image_id,
                                group=gkey, dist=dist, source=src, path=path,
                                box_idx=box_idx, is_center=is_center,
                                excluded=excluded, train_use=train_use))

    # center crop: training positive (if not excluded); always eval
    center_train = (label == 0 and not is_meglass) or (label == 1 and not excluded) \
        or (label == 0 and is_meglass)
    add(center_rgb, -1, True, center_train)

    # grid crops (skip for low-res MeGlass; skip for positives to avoid label noise)
    if is_meglass:
        return
    grid = c.grid[side]
    for gi, rgb in enumerate(grid):
        # negatives: every grid position is a label-0 hard negative (train + eval)
        # positives: grid crops are eval-only (module may be absent off-center)
        add(rgb, gi, False, train_use=(label == 0))


def _crops_for_file(entry, cfg, excl, verbose):
    """Worker: load one image, localize, and return its CornerItems (or [])."""
    folder, label, dist, path = entry
    try:
        img = cv2.imread(path)
        if img is None:
            from pipeline.preprocess import load_image
            img = load_image(path)             # Pillow fallback (RGBA/odd PNGs)
        c = corners.extract(img, cfg)
        if c is None:                          # no face -> not a worn shot, skip
            return []
        src = _source_of(folder, path)
        gkey = _group_key(src, path)
        image_id = src + ":" + os.path.splitext(os.path.basename(path))[0]
        local: list = []
        for side in ("L", "R"):
            _emit_side(local, c, side, label, image_id, gkey, dist, src, path, excl)
        return local
    except Exception as e:  # noqa: BLE001
        if verbose:
            print("skip", path, e)
        return []


def load_corner_dataset(cfg: Config, verbose=False):
    """Localize the face + candidate grid, emit one CornerItem per candidate crop.
    Exact-duplicate files (same md5) load once (grouping is per-file, so dups would
    straddle CV folds). Returns list[CornerItem]."""
    excl = _load_exclusions()

    # Phase 1 (serial): md5-dedup in deterministic SOURCES/_files order — first-seen
    # file wins, later duplicates dropped. Order is preserved so the CV split and the
    # seeded sampler see the identical item sequence regardless of parallelism below.
    seen_md5: dict = {}
    work, ndup = [], 0
    for folder, label, dist in SOURCES:
        if not os.path.isdir(folder):
            continue
        for path in _files(folder):
            with open(path, "rb") as fh:
                digest = hashlib.md5(fh.read()).hexdigest()
            if digest in seen_md5:
                ndup += 1
                if verbose:
                    print("dup", path, "==", seen_md5[digest])
                continue
            seen_md5[digest] = path
            work.append((folder, label, dist, path))

    # Phase 2 (parallel): the heavy Haar/localization per file, across cores.
    workers = max(1, min(LOAD_WORKERS, len(work)))
    if workers == 1:
        results = [_crops_for_file(e, cfg, excl, verbose) for e in work]
    else:
        fn = partial(_crops_for_file, cfg=cfg, excl=excl, verbose=verbose)
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_worker_init) as ex:
            results = list(ex.map(fn, work, chunksize=8))   # map preserves order
    items = [it for local in results for it in local]
    if ndup:
        print(f"dedupe: skipped {ndup} exact-duplicate files")
    ntrain = sum(1 for it in items if it.train_use)
    nx = sum(1 for it in items if it.excluded and it.is_center)
    print(f"corner items: {len(items)} crops ({ntrain} training) | "
          f"excluded sides: {nx}")
    return items


if __name__ == "__main__":
    it = load_corner_dataset(Config(), verbose=True)
    imgs = {x.image_id for x in it}
    posimg = {x.image_id for x in it if x.label == 1}
    tr = [x for x in it if x.train_use]
    print(f"{len(it)} crops from {len(imgs)} images ({len(posimg)} rayban) | "
          f"train: {sum(1 for x in tr if x.label==1)} pos / {sum(1 for x in tr if x.label==0)} neg | "
          f"groups={len({x.group for x in it})}")
