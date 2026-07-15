#!/usr/bin/env python3
"""Evaluate the trained two-class detector and compare against the old baseline.

The model emits rayban_meta (cls 0) + glasses (cls 1) boxes; the verdict is
three-way:  META if rayban >= t1, else NORMAL if glasses >= t2, else NOGLASSES.

Outputs:
  1. Ultralytics validator metrics (YOLO weights only).
  2. A whole-folder sweep over the original image folders -> per-image CSV
     (rayban_score, glasses_score, 3-way predicted, rayban-binary outcome),
     still directly comparable to the old detection_accuracy.csv baseline.
     --sweep picks t1 FP-first (as before), then a second 1-D sweep picks t2
     minimizing (normal->NOGLASSES) + (noglasses->NORMAL) errors.
  3. A 3x3 confusion matrix, with the no-glasses accuracy ALSO reported on the
     held-out val subset only (most of the hirepro folder was trained on).
  4. The failure-mode regression check against the old detection_accuracy.csv.

    python eval_detector.py --weights runs/rayban_yolox/best_ckpt.pth --sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
from infer import build_detector, pick_device, _iter_images, _DEFAULT_YOLO  # noqa: E402

# (folder, true_type). Mirrors datasets.SOURCES minus the 500 MeGlass bulk. The
# hirepro folders are mostly TRAINING data — their rows feed the t2 sweep and the
# no-glasses check (val-only numbers are reported separately as the honest metric);
# the rayban precision/recall stays comparable to the old baseline folders.
EVAL_FOLDERS = [
    ("ray_ban_face", "rayban"),
    ("ray_ban_frame", "rayban"),
    ("actual_rayban_cases", "rayban"),
    ("normal_glassess", "normal"),
    ("normal_frame", "normal"),
    ("hirepro_images_normal/8-july/glasses with good lighting ", "normal"),
    ("hirepro_images_normal/8-july/glasses with normal-low light", "normal"),
    ("hirepro_images_normal/8-july/without glasses", "noglasses"),
]
OLD_CSV = os.path.join(_REPO, "detection_accuracy.csv")
OUT_CSV = os.path.join(_REPO, "detection_accuracy_yolo.csv")
BOXES_JSON = os.path.join(_REPO, "data", "boxes", "boxes.json")


def outcome(true_type, predicted):
    """Rayban-binary outcome (compat with the old baseline CSV)."""
    pos = predicted == "META"
    if true_type == "rayban":
        return "TP" if pos else "FN"
    return "FP" if pos else "TN"


def predict3(rb, gl, t1, t2):
    if rb >= t1:
        return "META"
    if gl >= t2:
        return "NORMAL"
    return "NOGLASSES"


def load_val_groups():
    """Group keys of records held out in the val split, per detector class, from
    data/boxes/boxes.json — used to report honest (never-trained-on) numbers."""
    out = {"noglasses": set(), "glasses": set()}
    if not os.path.isfile(BOXES_JSON):
        return out
    for r in json.load(open(BOXES_JSON))["images"]:
        if r.get("split") != "val":
            continue
        cls = r.get("cls", "rayban_meta" if r.get("box") else None)
        if cls is None:
            out["noglasses"].add(r["group"])
        elif cls == "glasses":
            out["glasses"].add(r["group"])
    return out


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
    ap.add_argument("--conf", type=float, default=0.4,
                    help="t1: min rayban score for META")
    ap.add_argument("--conf-glasses", type=float, default=0.35,
                    help="t2: min glasses score for NORMAL (else NOGLASSES)")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep t1 0.20-0.85 (FP-first pick, as before), then sweep "
                         "t2 minimizing normal<->noglasses confusion")
    ap.add_argument("--max-fp", type=int, default=1,
                    help="FP-first policy: recommend the lowest t1 whose false "
                         "positives on non-rayban stay <= this (default 1)")
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
    val_groups = load_val_groups()

    # Detect ONCE at a low floor and cache the top score PER CLASS per image; the
    # operating-point tally and both --sweep tables are then computed in Python
    # (no re-running the model per threshold).
    floor = 0.05 if args.sweep else min(args.conf, args.conf_glasses)
    items = []  # (true_type, rb_top, gl_top, name, prev_outcome, rel_path, group)
    for folder, true_type in EVAL_FOLDERS:
        abs_folder = os.path.join(_REPO, folder)
        if not os.path.isdir(abs_folder):
            continue
        # the SOURCES folder (group keys are folder:stem; hirepro sub-folders all
        # live under one source)
        src = ("hirepro_images_normal/8-july"
               if folder.startswith("hirepro_images_normal/8-july") else folder)
        for path in _iter_images(abs_folder):
            boxes = det.detect(path, floor)
            rb = max((b[4] for b in boxes if b[5] == 0), default=0.0)
            gl = max((b[4] for b in boxes if b[5] == 1), default=0.0)
            name = os.path.basename(path)
            group = src + ":" + os.path.splitext(name)[0]
            items.append((true_type, rb, gl, name, old.get(name, ""),
                          os.path.relpath(path, _REPO), group))

    def tally_at(t1):
        """Rayban-binary tally (baseline-comparable): META vs not."""
        t = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        for true_type, rb, *_ in items:
            pred = "META" if rb >= t1 else "NONE"
            t[outcome(true_type, pred)] += 1
        return t

    def confusion(t1, t2):
        """3x3 confusion: true_type x predicted."""
        c = {tt: {"META": 0, "NORMAL": 0, "NOGLASSES": 0}
             for tt in ("rayban", "normal", "noglasses")}
        for true_type, rb, gl, *_ in items:
            c[true_type][predict3(rb, gl, t1, t2)] += 1
        return c

    def metrics(t):
        tp, fp, tn, fn = t["TP"], t["FP"], t["TN"], t["FN"]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        a = (tp + tn) / max(1, tp + fp + tn + fn)
        return p, r, a

    confs = [round(0.20 + 0.05 * i, 2) for i in range(14)]  # 0.20..0.85

    # ---- t1 sweep + FP-first recommendation (unchanged policy) ----
    if args.sweep:
        print("=== t1 sweep: rayban threshold (all eval folders) ===")
        print("  conf   TP  FP  TN  FN   prec   recall")
        best_conf = None
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
        print(f"\n  recommended t1 = {best_conf:.2f} "
              f"(highest recall with FP <= {args.max_fp} non-rayban)")
        args.conf = best_conf

        # ---- t2 sweep: minimize normal<->noglasses confusion (given t1) ----
        print("\n=== t2 sweep: glasses threshold (given t1) ===")
        print("  conf   normal->NOGLASSES  noglasses->NORMAL  errors")
        best_t2, best_err = None, None
        for c in confs:
            miss = sum(1 for tt, rb, gl, *_ in items
                       if tt == "normal" and rb < args.conf and gl < c)
            ghost = sum(1 for tt, rb, gl, *_ in items
                        if tt == "noglasses" and rb < args.conf and gl >= c)
            err = miss + ghost
            print(f"  {c:.2f}        {miss:3d}                {ghost:3d}"
                  f"           {err:3d}")
            if best_err is None or err < best_err:
                best_t2, best_err = c, err
        print(f"\n  recommended t2 = {best_t2:.2f} ({best_err} errors)")
        args.conf_glasses = best_t2

    # ---- operating-point tally + CSV + regression ----
    op = tally_at(args.conf)
    old_fn_rec = old_fn_tot = old_fp_fix = old_fp_tot = 0
    rows = []
    for true_type, rb, gl, name, prev, rel, group in items:
        pred = predict3(rb, gl, args.conf, args.conf_glasses)
        oc = outcome(true_type, pred)
        if prev == "FN":
            old_fn_tot += 1
            old_fn_rec += (pred == "META")
        elif prev == "FP":
            old_fp_tot += 1
            old_fp_fix += (pred != "META")
        rows.append({"file_path": rel, "image_name": name, "true_type": true_type,
                     "rayban_score": f"{rb:.3f}", "glasses_score": f"{gl:.3f}",
                     "predicted": pred, "outcome": oc, "old_outcome": prev})

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file_path", "image_name", "true_type",
                                           "rayban_score", "glasses_score",
                                           "predicted", "outcome", "old_outcome"])
        w.writeheader()
        w.writerows(rows)

    p, r, a = metrics(op)
    print(f"\n=== operating point (t1={args.conf:.2f}, t2={args.conf_glasses:.2f}) "
          f"-> {os.path.basename(out_csv)} ===")
    print(f"  images={len(rows)}  rayban-binary: TP={op['TP']} FP={op['FP']} "
          f"TN={op['TN']} FN={op['FN']}")
    print(f"  rayban precision={p:.3f}  recall={r:.3f}  accuracy={a:.3f}")
    print(f"  OLD baseline: TP=41 FP=18 TN=51 FN=17  (recall 0.707, precision 0.695)")

    conf3 = confusion(args.conf, args.conf_glasses)
    print("\n=== 3-way confusion (rows=truth, cols=predicted) ===")
    print("  truth\\pred    META  NORMAL  NOGLASSES")
    for tt in ("rayban", "normal", "noglasses"):
        c = conf3[tt]
        print(f"  {tt:<11s} {c['META']:5d} {c['NORMAL']:7d} {c['NOGLASSES']:10d}")

    # Honest no-glasses number: only the held-out val images (the rest of the
    # hirepro folder was trained on as background).
    ng = [(tt, rb, gl, g) for tt, rb, gl, _n, _p, _r, g in items
          if tt == "noglasses"]
    ng_val = [x for x in ng if x[3] in val_groups["noglasses"]]
    if ng:
        ok = sum(1 for tt, rb, gl, _ in ng
                 if predict3(rb, gl, args.conf, args.conf_glasses) == "NOGLASSES")
        print(f"\n=== no-glasses accuracy ===")
        print(f"  whole folder     : {ok}/{len(ng)}  "
              f"(CAUTION: most were in TRAIN as background)")
        if ng_val:
            okv = sum(1 for tt, rb, gl, _ in ng_val
                      if predict3(rb, gl, args.conf, args.conf_glasses) == "NOGLASSES")
            print(f"  held-out val only: {okv}/{len(ng_val)}  <- honest number "
                  f"(small set; collect more bare-face photos to data/noglasses/)")
        else:
            print("  (no val-split records found in data/boxes/boxes.json — "
                  "run tools/bootstrap_boxes.py first for the honest split)")

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
