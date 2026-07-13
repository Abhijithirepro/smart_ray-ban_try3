#!/usr/bin/env python3
"""Train the Ray-Ban Meta whole-glasses detector.

Two backends share one CLI:

  --model yolo    (default)  Ultralytics YOLO12s on data/rayban_yolo/. Fast; trains
                             on Apple MPS in minutes. The practical default here.
  --model rfdetr             RF-DETR (transformer, COCO-pretrained) on
                             data/rayban_coco/. Higher accuracy on custom data but
                             slow without a CUDA GPU — recommended for the final run
                             on a GPU box / Colab.

Both train on the SAME underlying data (real + augmented, exported by
tools/export_dataset.py). Weights + curves land under runs/.

    python train_detector.py --model yolo --epochs 120
    python train_detector.py --model yolo --epochs 3        # smoke test
    python train_detector.py --model rfdetr --epochs 60 --batch 4
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_yolo(args):
    from ultralytics import YOLO
    data = os.path.join(_REPO, "data", "rayban_yolo", "rayban.yaml")
    if not os.path.isfile(data):
        raise SystemExit("missing data/rayban_yolo — run "
                         "tools/export_dataset.py --format yolo")

    # Resume an interrupted run from its last checkpoint (keeps the LR schedule).
    if args.resume:
        last = os.path.join(_REPO, "runs", "rayban_yolo", "weights", "last.pt")
        if not os.path.isfile(last):
            raise SystemExit(f"--resume: no checkpoint at {last}")
        print(f"RESUME from {last}")
        YOLO(last).train(resume=True)
        best = os.path.join(_REPO, "runs", "rayban_yolo", "weights", "best.pt")
        print(f"\ndone. best weights: {best}")
        return best

    weights = args.weights or "yolo12s.pt"
    device = args.device or pick_device()
    print(f"YOLO train: {weights}  data={data}  device={device}  "
          f"imgsz={args.imgsz}  epochs={args.epochs}")
    model = YOLO(weights)
    model.train(
        data=data, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=device, patience=args.patience,
        project=os.path.join(_REPO, "runs"), name="rayban_yolo",
        exist_ok=True, seed=0, cache="ram",
        # single big object, small real set -> keep strong aug on; our offline aug
        # already injected distance/occlusion/glare, so this is complementary.
        mosaic=1.0, close_mosaic=10, mixup=0.1, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=8.0, translate=0.1, scale=0.5, fliplr=0.5,
    )
    best = os.path.join(_REPO, "runs", "rayban_yolo", "weights", "best.pt")
    print(f"\ndone. best weights: {best}")
    return best


def train_rfdetr(args):
    dataset = os.path.join(_REPO, "data", "rayban_coco")
    if not os.path.isdir(dataset):
        raise SystemExit("missing data/rayban_coco — run "
                         "tools/export_dataset.py --format coco")
    try:
        from rfdetr import RFDETRSmall
    except ImportError:
        raise SystemExit("rfdetr not installed: .venv-train/bin/python -m pip install rfdetr")
    out = os.path.join(_REPO, "runs", "rayban_rfdetr")
    print(f"RF-DETR train: dataset={dataset}  out={out}  epochs={args.epochs}")
    if pick_device() != "cuda":
        print("WARNING: no CUDA GPU — RF-DETR training will be slow. Consider "
              "--model yolo here, or run this on a GPU/Colab.", file=sys.stderr)
    model = RFDETRSmall()
    model.train(
        dataset_dir=dataset, epochs=args.epochs,
        batch_size=args.batch, grad_accum_steps=max(1, 16 // max(1, args.batch)),
        lr=1e-4, output_dir=out,
    )
    print(f"\ndone. checkpoints under: {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["yolo", "rfdetr"], default="yolo")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=960,
                    help="YOLO input size; 960 keeps the small module resolvable")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--weights", default=None, help="YOLO base weights (default yolo12s.pt)")
    ap.add_argument("--resume", action="store_true",
                    help="resume the interrupted runs/rayban_yolo run from last.pt")
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default auto)")
    args = ap.parse_args(argv)

    if args.model == "yolo":
        train_yolo(args)
    else:
        train_rfdetr(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
