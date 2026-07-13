#!/usr/bin/env python3
"""Re-derive the MAYBE (review) tier threshold WITHOUT retraining.

tc_hi (the confident-YES bar) is fixed by construction: it clears every in-domain
worn normal's second-best corner, so no normal can ever reach a confident YES
(zero YES-FP). tc_maybe (the review band) is a policy operating point — how many
hard normals you're willing to send to review in exchange for catching more
Ray-Bans as MAYBE. train_corner_clf derives it from a review budget; the default
(0.10) is often too tight for a thick-frame-heavy negative set and collapses the
band onto tc_hi (tc_maybe == tc_hi), which is what makes recall look poor.

This reads the existing OOF report + trained checkpoint, re-derives tc_maybe at a
chosen review budget using the SAME shipped logic (train_corner_clf.pick_tier_
thresholds), patches models/corner_clf.pt in place, and rewrites the report's
tc_maybe / tier_counts / per-image tier. tc_hi is left untouched. No retrain.

    python tools/retune_tier_maybe.py --budget 0.25

Then re-export to the browser: python tools/export_corner_onnx.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import train_corner_clf as TC

REPORT = "models/corner_oof_report.json"
CKPT = "models/corner_clf.pt"


def _imgs_from_report(report):
    """Reconstruct the {image_id: {...}} dict pick_tier_thresholds expects from the
    saved per-image report rows (corner_probs -> probs)."""
    imgs = {}
    for d in report["images"]:
        imgs[d["image_id"]] = {"probs": d["corner_probs"], "label": d["label"],
                               "dist": d["dist"], "source": d["source"]}
    return imgs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=0.25,
                    help="MAYBE review budget: max fraction of in-domain worn "
                         "normals allowed to be flagged for review (default 0.25)")
    args = ap.parse_args(argv)

    report = json.load(open(REPORT))
    imgs = _imgs_from_report(report)

    tc_maybe_old = report["tc_maybe"]
    # Re-derive from the report's (rounded-to-4dp) probs so the tier counts below
    # are self-consistent with those same rounded probs — comparing a rounded prob
    # against the checkpoint's full-precision tc_hi can spuriously flip the single
    # normal that SET tc_hi to YES (a display artifact, not a real FP). We only patch
    # tc_maybe into the checkpoint; its full-precision tc_hi is left untouched, so
    # the shipped zero-YES-FP guarantee is unaffected regardless of this value.
    tc_hi, tc_maybe = TC.pick_tier_thresholds(imgs, review_budget=args.budget)
    report["tc_hi"] = tc_hi

    # recompute tier verdicts + per-group counts with the new tc_maybe
    for d in report["images"]:
        d["tier"] = TC.tier_verdict(d["corner_probs"], tc_hi, tc_maybe)
    counts = {}
    rows = (("RAYBAN worn", lambda d: d["label"] == 1 and d["dist"] == "real"),
            ("NORMAL in-domain", lambda d: d["label"] == 0 and d["dist"] == "real"
             and "meglass" not in d["source"]),
            ("NORMAL meglass", lambda d: "meglass" in d["source"]))
    for name, pred in rows:
        g = [d for d in report["images"] if pred(d)]
        y = sum(1 for d in g if d["tier"] == "YES")
        mb = sum(1 for d in g if d["tier"] == "MAYBE")
        counts[name] = {"YES": y, "MAYBE": mb, "NO": len(g) - y - mb, "total": len(g)}

    report["tc_maybe"] = tc_maybe
    report["tier_counts"] = counts
    with open(REPORT, "w") as fh:
        json.dump(report, fh, indent=2)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ck["tc_maybe"] = tc_maybe
    torch.save(ck, CKPT)

    print(f"budget={args.budget}  tc_hi={tc_hi:.3f} (unchanged)  "
          f"tc_maybe {tc_maybe_old:.3f} -> {tc_maybe:.3f}")
    for name in ("RAYBAN worn", "NORMAL in-domain", "NORMAL meglass"):
        c = counts[name]
        extra = (f"  flagged={100*(c['YES']+c['MAYBE'])/max(1,c['total']):.0f}%"
                 if "RAYBAN" in name else "")
        print(f"  {name:18} YES={c['YES']:<3} MAYBE={c['MAYBE']:<3} "
              f"NO={c['NO']:<4} of {c['total']}{extra}")
    print(f"patched {CKPT} and {REPORT} — now run tools/export_corner_onnx.py")


if __name__ == "__main__":
    main()
