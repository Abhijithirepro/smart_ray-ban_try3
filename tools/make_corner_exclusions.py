#!/usr/bin/env python3
"""Build the corner-label exclusion list from out-of-fold probabilities.

Root cause this fixes: datasets_corners.py labels BOTH corners of every Ray-Ban
photo "camera module", but on side-profile / one-hand-occluded shots one corner
shows no module at all (~24% of positive corners scored <0.2 OOF). Training on
those poisoned positives blurs the model's module concept until thick normal
frame corners start firing.

Rule (validated): for each POSITIVE image, exclude corner i from TRAINING iff
    prob[i] < CUT (0.3)   and   corner i is not the image's max corner.
The max corner is always kept, so no image loses all positive signal. Excluded
corners are dropped from training only — never taught as negative — and still
scored at eval time. The 0.3 cut (not 0.5) keeps the 0.3-0.5 borderline-visible
hard positives in training.

Bootstraps from the EXISTING models/corner_oof_report.json — corner visibility
is a property of the image, not of the model run, so no fresh retrain is needed
to build the list.

    python tools/make_corner_exclusions.py [--cut 0.3]

Outputs:
  models/corner_exclusions.json          ["<image_id>:<side>", ...]
  debug/excluded_corners/{prob}__{image}__{side}.png   (visual audit)
Optional rescue list models/corner_keep.txt: one "<image_id>:<side>" per line
that must never be excluded (audited as wrongly excluded by a human).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPORT = "models/corner_oof_report.json"
OUT = "models/corner_exclusions.json"
KEEP = "models/corner_keep.txt"
DUMP_DIR = os.path.join("debug", "excluded_corners")
SIDES = ("L", "R")   # corner_probs order in the report is [L, R]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cut", type=float, default=0.3,
                    help="exclude a non-max positive corner below this OOF prob")
    ap.add_argument("--no-dump", action="store_true", help="skip the crop dump")
    args = ap.parse_args(argv)

    if not os.path.exists(REPORT):
        print(f"error: {REPORT} not found — run train_corner_clf.py once first",
              file=sys.stderr)
        return 2
    report = json.load(open(REPORT))

    keep = set()
    if os.path.exists(KEEP):
        keep = {ln.strip() for ln in open(KEEP) if ln.strip()}
        print(f"rescue list: {len(keep)} entries from {KEEP}")

    excl = []
    for img in report["images"]:
        if img["label"] != 1:
            continue
        probs = img["corner_probs"]
        mx = max(probs)
        for i, p in enumerate(probs):
            key = f"{img['image_id']}:{SIDES[i]}"
            if p < args.cut and p != mx and key not in keep:
                excl.append({"key": key, "prob": p, "path": img["path"]})

    with open(OUT, "w") as fh:
        json.dump({"cut": args.cut, "source_report": REPORT,
                   "excluded": [e["key"] for e in excl]}, fh, indent=2)
    npos = sum(1 for i in report["images"] if i["label"] == 1)
    print(f"excluded {len(excl)} of {2*npos} positive corners "
          f"(cut={args.cut}) -> {OUT}")

    if not args.no_dump and excl:
        # regenerate the actual crops for visual audit
        from config import Config
        from datasets_corners import load_corner_dataset
        from PIL import Image
        os.makedirs(DUMP_DIR, exist_ok=True)
        wanted = {e["key"]: e["prob"] for e in excl}
        items = load_corner_dataset(Config())
        n = 0
        for it in items:
            key = f"{it.image_id}:{it.side}"
            if key in wanted:
                base = os.path.splitext(os.path.basename(it.path))[0]
                Image.fromarray(it.rgb).save(os.path.join(
                    DUMP_DIR, f"{wanted[key]:.3f}__{base}__{it.side}.png"))
                n += 1
        print(f"dumped {n} excluded crops to {DUMP_DIR}/ — glance through them; "
              f"rescue any wrongly-excluded via {KEEP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
