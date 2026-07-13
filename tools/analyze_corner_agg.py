"""Offline sweep of per-side grid aggregation statistics on the corner OOF report.

Reads the per-box OOF vectors (boxes_L/boxes_R, grid boxes only) captured by
train_corner_clf and, for each candidate statistic, runs the REAL three-tier
threshold logic (pick_tier_thresholds / tier_verdict) to report: tc_hi, tc_maybe,
confident-YES FP, normal soft-MAYBE, Ray-Ban YES + flagged%, plus a PER-FOLD
breakdown (the anti-noise-fit guardrail). Also prints a per-box spurious-rate /
unique-recovery table to inform any grid trim. No retrain — pick the statistic
here, then ship it via train_corner_clf.AGGREGATE.

    python tools/analyze_corner_agg.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPORT = "models/corner_oof_report.json"
MEGLASS = "data/normal/meglass"


def _stat(boxes, name):
    """Per-side score from the sorted-desc grid-box vector `boxes`."""
    if not boxes:
        return 0.0
    if name == "max":
        return boxes[0]
    if name == "second_high":
        return boxes[1] if len(boxes) >= 2 else boxes[0]
    if name == "top2_mean":
        return (boxes[0] + (boxes[1] if len(boxes) >= 2 else boxes[0])) / 2.0
    raise ValueError(name)


def _sides(img, name):
    return [_stat(img.get("boxes_L", []), name), _stat(img.get("boxes_R", []), name)]


def pick_tiers(negs_probs, review_budget=0.10):
    """Mirror of train_corner_clf.pick_tier_thresholds on a list of [L,R] normal
    scores. tc_hi clears every normal's weaker side; tc_maybe keeps normal MAYBE
    within budget. Returns (tc_hi, tc_maybe)."""
    if not negs_probs:
        return 0.9, 0.5
    second = [min(p) for p in negs_probs]           # weaker side
    tc_hi = min(float(np.nextafter(max(second), 1.0)), 0.999)
    budget = int(np.ceil(review_budget * len(negs_probs)))
    neg_max = np.array([max(p) for p in negs_probs])
    tc_maybe = tc_hi
    for t in np.linspace(0.5, tc_hi, 200):
        if int((neg_max >= t).sum()) <= budget:
            tc_maybe = float(t); break
    return tc_hi, tc_maybe


def verdict(pr, tc_hi, tc_maybe):
    if sum(1 for p in pr if p >= tc_hi) >= 2:
        return "YES"
    if any(p >= tc_maybe for p in pr):
        return "MAYBE"
    return "NO"


def evaluate(pos, negi, name):
    negp = [_sides(i, name) for i in negi]
    posp = [_sides(i, name) for i in pos]
    tc_hi, tc_maybe = pick_tiers(negp)
    pv = [verdict(p, tc_hi, tc_maybe) for p in posp]
    nv = [verdict(p, tc_hi, tc_maybe) for p in negp]
    return {
        "tc_hi": tc_hi, "tc_maybe": tc_maybe,
        "yes": pv.count("YES"), "maybe": pv.count("MAYBE"), "npos": len(pv),
        "flagged": (pv.count("YES") + pv.count("MAYBE")) / max(1, len(pv)),
        "neg_yes": nv.count("YES"), "neg_maybe": nv.count("MAYBE"), "nneg": len(nv),
    }


def main():
    if not os.path.exists(REPORT):
        print("no", REPORT, "- run train_corner_clf.py first", file=sys.stderr)
        return 2
    r = json.load(open(REPORT))
    if "boxes_L" not in (r["images"][0] if r["images"] else {}):
        print("report has no per-box vectors — retrain with the instrumented "
              "train_corner_clf.py first", file=sys.stderr)
        return 2
    imgs = [i for i in r["images"] if i["dist"] == "real"]
    pos = [i for i in imgs if i["label"] == 1]
    negi = [i for i in imgs if i["label"] == 0 and i["source"] != MEGLASS]
    print(f"worn images: {len(pos)} rayban, {len(negi)} in-domain normal")

    stats = ["max", "second_high", "top2_mean"]
    print("\n=== POOLED (all folds) ===")
    print(f"  {'stat':12} tc_hi  tc_mb  YES  flagged  | normal YES  MAYBE")
    for s in stats:
        m = evaluate(pos, negi, s)
        gap = "COLLAPSE" if abs(m["tc_maybe"] - m["tc_hi"]) < 1e-6 else "ok"
        print(f"  {s:12} {m['tc_hi']:.3f}  {m['tc_maybe']:.3f}  {m['yes']:>3}  "
              f"{m['flagged']*100:5.0f}%   | {m['neg_yes']:>3}  {m['neg_maybe']:>3}/{m['nneg']}  ({gap})")

    # ---- per-fold guardrail: normal-MAYBE per fold for each stat ----
    from train_corner_clf import grouped_folds
    from datasets_corners import load_corner_dataset
    from config import Config
    items = load_corner_dataset(Config())
    fold_of = grouped_folds(items, r.get("kfolds", 4))
    kf = r.get("kfolds", 4)
    print(f"\n=== PER-FOLD normal soft-MAYBE (guardrail: want the winner low in >=3/{kf}) ===")
    print(f"  {'stat':12} " + " ".join(f"f{k}" for k in range(kf)) + "   (thresholds from pooled normals)")
    for s in stats:
        # use pooled thresholds (as shipped), count normal MAYBE within each fold
        tc_hi, tc_maybe = pick_tiers([_sides(i, s) for i in negi])
        per = []
        for k in range(kf):
            fn = [i for i in negi if fold_of.get(i["group"]) == k]
            nm = sum(1 for i in fn if verdict(_sides(i, s), tc_hi, tc_maybe) == "MAYBE")
            per.append(f"{nm:>2}")
        print(f"  {s:12} " + " ".join(per))

    # ---- per-box spurious rate / unique recovery (grid-trim guidance) ----
    print("\n=== per-box index: normal-argmax rate | positive unique-recovery ===")
    nb = max(len(i.get("boxes_L", [])) for i in imgs) if imgs else 0
    # boxes stored sorted-desc, so index != geometry index; report the count of
    # sides where the max box is 'lone' (>=0.9 with 2nd <0.5) as a spuriousness proxy
    lone_norm = sum(1 for i in negi for b in ([i.get("boxes_L", []), i.get("boxes_R", [])])
                    if b and b[0] >= 0.9 and (len(b) < 2 or b[1] < 0.5))
    lone_pos = sum(1 for i in pos for b in ([i.get("boxes_L", []), i.get("boxes_R", [])])
                   if b and b[0] >= 0.9 and (len(b) < 2 or b[1] < 0.5))
    print(f"  'lone high box' sides (max>=0.9 & 2nd<0.5): normal={lone_norm}  rayban={lone_pos}")
    print("  (many lone-high normal sides + few lone-high rayban => 2nd-highest is the right cut)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
