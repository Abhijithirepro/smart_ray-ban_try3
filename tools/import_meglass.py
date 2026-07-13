#!/usr/bin/env python3
"""One-time importer: MeGlass eyeglass faces -> data/normal/meglass/ hard negatives.

MeGlass (github.com/cleardusk/MeGlass) is thousands of celebrity faces wearing
ORDINARY eyeglasses — i.e. perfect bulk hard negatives for the Ray-Ban-vs-normal
CNN (they are exactly the "dark frame on a face" images the old detector
false-flagged). We add a face-gated, per-identity sample of them so the classifier
learns "ordinary frame on a face != Ray-Ban Meta" at scale, driving worn-case
false positives toward zero.

    python tools/import_meglass.py --src /path/to/MeGlass_ori --n 500 --seed 0

What it does:
  * Selects the EYEGLASSES-ON subset (from meta.txt if present; glasses-off faces
    are an easier, different negative and would dilute the signal).
  * FACE-GATES each candidate: keeps only images where pipeline/region localizes a
    "worn" region (a genuine eye-line glasses band) — not a center-crop fallback of
    a random photo, which would inject label noise.
  * Samples ONE image per identity (MegaFace id = string before the 2nd '@'), so
    no identity is over-represented and CV folds stay clean.
  * Copies originals (byte-for-byte, original names) into data/normal/meglass/ and
    writes manifest.json for provenance/reproducibility.

Use MeGlass_ori (full images). The 120x120 aligned crops are too small: after
region-cropping they fall well below the 160px CNN input.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from datasets import meglass_identity
from pipeline import preprocess, region

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEST = os.path.join("data", "normal", "meglass")


def _all_images(root):
    out = []
    for ext in EXTS:
        out += glob.glob(os.path.join(root, "**", "*" + ext), recursive=True)
        out += glob.glob(os.path.join(root, "*" + ext))
    return sorted(set(out))


def _find_meta(src, explicit):
    if explicit:
        return explicit
    for cand in (os.path.join(src, "meta.txt"),
                 os.path.join(os.path.dirname(src.rstrip("/")), "meta.txt")):
        if os.path.isfile(cand):
            return cand
    return None


def _glasses_on_names(meta_path):
    """Set of basenames labelled eyeglasses-ON, or None if no usable meta.

    Handles the common 'name label' pair format (1=black-eyeglass, 0=no) and a
    names-only fallback (every listed name is a glasses image)."""
    if not meta_path or not os.path.isfile(meta_path):
        return None
    names, paired = set(), False
    with open(meta_path, "r") as fh:
        for line in fh:
            toks = line.split()
            if not toks:
                continue
            name = os.path.basename(toks[0])
            if len(toks) >= 2 and toks[-1].strip() in ("0", "1"):
                paired = True
                if toks[-1].strip() == "1":
                    names.add(name)
            else:
                names.add(name)         # names-only line
    if not paired and names:
        print(f"  meta.txt has no 0/1 labels — treating all {len(names)} listed "
              f"names as glasses-on")
    return names if names else None


def _face_gated(path, cfg):
    """True if region localization finds a WORN glasses band (not a fallback)."""
    try:
        fr = preprocess.preprocess(path, cfg)
        reg = region.locate_region(fr, cfg)
        return reg.method == "worn" and reg.w >= 20 and reg.h >= 20
    except Exception:  # noqa: BLE001
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="extracted MeGlass_ori directory")
    ap.add_argument("--meta", help="path to meta.txt (auto-detected in --src if omitted)")
    ap.add_argument("--n", type=int, default=500, help="images to import (1 per identity)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (reproducible sample)")
    ap.add_argument("--max-attempts-per-id", type=int, default=4,
                    help="face-gate at most this many images per identity")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.src):
        print(f"error: --src not a directory: {args.src}", file=sys.stderr)
        return 2

    import numpy as np
    rng = np.random.RandomState(args.seed)
    cfg = Config()

    imgs = _all_images(args.src)
    print(f"found {len(imgs)} images under {args.src}")
    if imgs:
        print("  example filenames:", [os.path.basename(p) for p in imgs[:5]])

    meta_path = _find_meta(args.src, args.meta)
    on = _glasses_on_names(meta_path)
    if on is not None:
        imgs = [p for p in imgs if os.path.basename(p) in on]
        print(f"eyeglasses-ON per meta.txt ({meta_path}): {len(imgs)} images")
    else:
        print("no meta.txt found — using ALL images under --src as glasses-on "
              "(point --src at the glasses-on set, or pass --meta)")

    # group candidates by identity
    by_id = {}
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        ident = meglass_identity(stem) or ("file:" + stem)
        by_id.setdefault(ident, []).append(p)
    idents = list(by_id.keys())
    rng.shuffle(idents)
    print(f"{len(idents)} identities; sampling up to {args.n} (1 image each, "
          f"face-gated) ...")

    os.makedirs(DEST, exist_ok=True)
    chosen = []           # (ident, src_path, dest_name)
    gated_out = 0
    for ident in idents:
        if len(chosen) >= args.n:
            break
        cands = by_id[ident]
        rng.shuffle(cands)
        for p in cands[:args.max_attempts_per_id]:
            if _face_gated(p, cfg):
                dest_name = os.path.basename(p)
                shutil.copy2(p, os.path.join(DEST, dest_name))
                chosen.append((ident, p, dest_name))
                break
        else:
            gated_out += 1
        if len(chosen) % 50 == 0 and chosen:
            print(f"  ... {len(chosen)} kept")

    manifest = {
        "source_dir": os.path.abspath(args.src),
        "meta": meta_path, "seed": args.seed, "requested": args.n,
        "identity_rule": "string before 2nd '@' (datasets.meglass_identity)",
        "kept": len(chosen),
        "identities_face_gated_out": gated_out,
        "items": [{"identity": i, "src": s, "file": d} for (i, s, d) in chosen],
    }
    with open(os.path.join(DEST, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nimported {len(chosen)} MeGlass hard negatives -> {DEST}/")
    print(f"  ({gated_out} identities had no face-gated image and were skipped)")
    print("wrote", os.path.join(DEST, "manifest.json"))
    print("next: python train_region_clf.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
