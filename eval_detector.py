#!/usr/bin/env python3
"""Evaluate the trained detector and compare against the old baseline.

Three outputs:
  1. Ultralytics validator metrics (precision / recall / mAP50) on the real val
     split in data/rayban_yolo (YOLO weights only).
  2. A whole-folder sweep over the original image folders -> detection_accuracy_yolo.csv
     (path, image_name, true_type, top_score, predicted, outcome), directly comparable
     to the existing detection_accuracy.csv baseline (TP=41 FP=18 TN=51 FN=17).
  3. A failure-mode regression check: recall on the 17 images the old system missed
     (FN) and false-positive rate on the 18 it wrongly flagged (FP), read from the
     old detection_accuracy.csv by image name.

    python eval_detector.py                             # yolo best.pt, conf 0.4
    python eval_detector.py --weights runs/.../best.pt --conf 0.35 --val-map
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
from infer import build_detector, pick_device, _iter_images, _DEFAULT_YOLO  # noqa: E402

# (folder, true_type). Mirrors datasets.SOURCES minus the 500 MeGlass bulk (kept
# comparable to the old baseline, which was scored on these in-domain folders).
EVAL_FOLDERS = [
    ("ray_ban_face", "rayban"),
    ("ray_ban_frame", "rayban"),
    ("actual_rayban_cases", "rayban"),
    ("normal_glassess", "normal"),
    ("normal_frame", "normal"),
]
OLD_CSV = os.path.join(_REPO, "detection_accuracy.csv")
OUT_CSV = os.path.join(_REPO, "detection_accuracy_yolo.csv")


def outcome(true_type, predicted):
    pos = predicted == "META"
    if true_type == "rayban":
        return "TP" if pos else "FN"
    return "FP" if pos else "TN"


def load_old_outcomes():
    """image_name -> old outcome (TP/FP/TN/FN) from detection_accuracy.csv."""
    old = {}
    if not os.path.isfile(OLD_CSV):
        return old
    with open(OLD_CSV) as fh:
        for row in csv.DictReader(fh):
            name = row.get("image_name")
            if name:
                old[name] = row.get("outcome", "").strip()
    return old


def val_map(weights, device):
    if not weights.endswith(".pt"):
        return
    yaml = os.path.join(_REPO, "data", "rayban_yolo", "rayban.yaml")
    if not os.path.isfile(yaml):
        return
    from ultralytics import YOLO
    print("=== Ultralytics validator (real val split) ===")
    m = YOLO(weights).val(data=yaml, device=device, verbose=False,
                          project=os.path.join(_REPO, "runs"), name="rayban_val",
                          exist_ok=True)
    b = m.box
    print(f"  precision={b.mp:.3f}  recall={b.mr:.3f}  "
          f"mAP50={b.map50:.3f}  mAP50-95={b.map:.3f}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=_DEFAULT_YOLO)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep conf 0.20-0.85 and print a precision/recall table, "
                         "then auto-pick the FP-first operating point")
    ap.add_argument("--max-fp", type=int, default=1,
                    help="FP-first policy: recommend the lowest conf whose false "
                         "positives on normals stay <= this (default 1)")
    ap.add_argument("--val-map", action="store_true",
                    help="also run the Ultralytics validator for mAP")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-csv", default=OUT_CSV,
                    help="per-image results CSV (default detection_accuracy_yolo.csv; "
                         "pass detection_accuracy_yolox.csv for the YOLOX run)")
    args = ap.parse_args(argv)
    out_csv = args.out_csv

    if not os.path.exists(args.weights):
        raise SystemExit(f"weights not found: {args.weights}")
    device = args.device or pick_device()
    det = build_detector(args.weights, device)

    if args.val_map:
        val_map(args.weights, device)

    old = load_old_outcomes()

    # Detect ONCE at a low floor and cache the top box score per image; both the
    # operating-point tally and the --sweep table are then computed in Python
    # (no re-running the model per threshold).
    floor = min(args.conf, 0.05) if args.sweep else args.conf
    items = []  # (true_type, top_score, name, prev_outcome, rel_path)
    for folder, true_type in EVAL_FOLDERS:
        abs_folder = os.path.join(_REPO, folder)
        if not os.path.isdir(abs_folder):
            continue
        for path in _iter_images(abs_folder):
            boxes = det.detect(path, floor)
            top = max((b[4] for b in boxes), default=0.0)
            name = os.path.basename(path)
            items.append((true_type, top, name, old.get(name, ""),
                          os.path.relpath(path, _REPO)))

    def tally_at(conf):
        t = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        for true_type, top, *_ in items:
            pred = "META" if top >= conf else "NONE"
            t[outcome(true_type, pred)] += 1
        return t

    def metrics(t):
        tp, fp, tn, fn = t["TP"], t["FP"], t["TN"], t["FN"]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        a = (tp + tn) / max(1, tp + fp + tn + fn)
        return p, r, a

    # ---- threshold sweep + FP-first recommendation ----
    if args.sweep:
        print("=== threshold sweep (in-domain folders) ===")
        print("  conf   TP  FP  TN  FN   prec   recall")
        best_conf = None
        confs = [round(0.20 + 0.05 * i, 2) for i in range(14)]  # 0.20..0.85
        for c in confs:
            t = tally_at(c)
            p, r, _ = metrics(t)
            print(f"  {c:.2f}  {t['TP']:3d} {t['FP']:3d} {t['TN']:3d} {t['FN']:3d}"
                  f"  {p:.3f}  {r:.3f}")
            # FP-first: lowest conf whose FP <= --max-fp (=> highest recall at that cap)
            if best_conf is None and t["FP"] <= args.max_fp:
                best_conf = c
        if best_conf is None:
            best_conf = confs[-1]
        print(f"\n  recommended operating conf = {best_conf:.2f} "
              f"(highest recall with FP <= {args.max_fp} normals)")
        args.conf = best_conf

    # ---- operating-point tally + CSV + regression ----
    op = tally_at(args.conf)
    old_fn_rec = old_fn_tot = old_fp_fix = old_fp_tot = 0
    rows = []
    for true_type, top, name, prev, rel in items:
        pred = "META" if top >= args.conf else "NONE"
        oc = outcome(true_type, pred)
        if prev == "FN":
            old_fn_tot += 1
            old_fn_rec += (pred == "META")
        elif prev == "FP":
            old_fp_tot += 1
            old_fp_fix += (pred == "NONE")
        rows.append({"file_path": rel, "image_name": name, "true_type": true_type,
                     "top_score": f"{top:.3f}", "predicted": pred,
                     "outcome": oc, "old_outcome": prev})

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file_path", "image_name", "true_type",
                                           "top_score", "predicted",
                                           "outcome", "old_outcome"])
        w.writeheader()
        w.writerows(rows)

    p, r, a = metrics(op)
    print(f"\n=== operating point (conf={args.conf:.2f}) -> {os.path.basename(out_csv)} ===")
    print(f"  images={len(rows)}  TP={op['TP']} FP={op['FP']} TN={op['TN']} FN={op['FN']}")
    print(f"  precision={p:.3f}  recall={r:.3f}  accuracy={a:.3f}")
    print(f"  OLD baseline: TP=41 FP=18 TN=51 FN=17  (recall 0.707, precision 0.695)")
    print("\n=== failure-mode regression (vs old detection_accuracy.csv) ===")
    if old_fn_tot:
        print(f"  old FALSE-NEGATIVES now recovered: {old_fn_rec}/{old_fn_tot}")
    if old_fp_tot:
        print(f"  old FALSE-POSITIVES now fixed:     {old_fp_fix}/{old_fp_tot}")
    if not old:
        print("  (detection_accuracy.csv not found — skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
