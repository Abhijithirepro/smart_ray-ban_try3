#!/usr/bin/env python3
"""Run the trained glasses detector on an image (or folder).

    python infer.py IMAGE [--weights runs/rayban_yolox/best_ckpt.pth] [--conf 0.4] [--save]
    python infer.py normal_glassess/ --save          # batch a whole folder

Two-class model (rayban_meta + glasses) -> three-way verdict per image:
    RAY-BAN META (0.87)    path     rayban box >= --conf
    NORMAL GLASSES (0.71)  path     else glasses box >= --conf-glasses
    NO GLASSES             path     else
With --save, writes an annotated copy (boxes + scores) to detected_image_draw/.

Works with YOLOX checkpoints (.pth under a yolox run dir), Ultralytics YOLO (.pt),
and RF-DETR if installed. Detectors return (x1, y1, x2, y2, score, cls) with
cls 0 = rayban_meta, 1 = glasses.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2

_REPO = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_YOLO = os.path.join(_REPO, "runs", "rayban_yolo", "weights", "best.pt")
_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
OUT_DIR = os.path.join(_REPO, "detected_image_draw")


def _iter_images(target):
    if os.path.isdir(target):
        for root, _, files in os.walk(target):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in _EXTS:
                    yield os.path.join(root, f)
    else:
        yield target


CLS_NAMES = ("rayban_meta", "glasses")
CLS_COLORS = ((0, 220, 60), (0, 165, 255))   # rayban green, glasses orange


def _annotate(img, boxes, out_path):
    for (x1, y1, x2, y2, score, cls) in boxes:
        c = int(cls) if int(cls) < len(CLS_NAMES) else 0
        color = CLS_COLORS[c]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        label = f"{CLS_NAMES[c]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (int(x1), int(y1) - th - 8),
                      (int(x1) + tw + 6, int(y1)), color, -1)
        cv2.putText(img, label, (int(x1) + 3, int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 40, 10), 2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)


class YoloDetector:
    def __init__(self, weights, device):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device

    def detect(self, path, conf):
        r = self.model.predict(path, conf=conf, device=self.device, verbose=False)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append((x1, y1, x2, y2, float(b.conf[0]), int(b.cls[0])))
        return out


class RFDetrDetector:
    def __init__(self, weights, device):
        from rfdetr import RFDETRSmall
        self.model = RFDETRSmall(pretrain_weights=weights)
        self.conf = 0.0

    def detect(self, path, conf):
        from PIL import Image
        import numpy as np
        det = self.model.predict(Image.open(path).convert("RGB"), threshold=conf)
        out = []
        xy = np.asarray(det.xyxy) if det.xyxy is not None else []
        cf = np.asarray(det.confidence) if det.confidence is not None else []
        cid = np.asarray(det.class_id) if getattr(det, "class_id", None) is not None \
            else np.zeros(len(cf))
        for (x1, y1, x2, y2), s, c in zip(xy, cf, cid):
            out.append((float(x1), float(y1), float(x2), float(y2),
                        float(s), int(c)))
        return out


_DEFAULT_YOLOX_EXP = os.path.join(_REPO, "exps", "rayban_yolox_nano.py")


class YoloxDetector:
    """Adapter for a YOLOX checkpoint (e.g. runs/rayban_yolox/best_ckpt.pth).

    Mirrors YoloDetector.detect: returns [(x1, y1, x2, y2, score), ...] in the
    ORIGINAL image's pixel coordinates. The architecture is rebuilt from the Exp
    file (default exps/rayban_yolox_nano.py, override via $YOLOX_EXP)."""

    def __init__(self, weights, device):
        import torch
        from yolox.exp import get_exp
        self.torch = torch
        exp = get_exp(os.environ.get("YOLOX_EXP", _DEFAULT_YOLOX_EXP), None)
        self.model = exp.get_model()
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.device = device if device in ("cuda", "mps") else "cpu"
        self.model.to(self.device)
        self.test_size = exp.test_size       # (640, 640)
        self.num_classes = exp.num_classes   # 2 (rayban_meta, glasses)
        self.nmsthre = exp.nmsthre           # 0.65

    def detect(self, path, conf):
        from yolox.data.data_augment import preproc
        from yolox.utils import postprocess
        img = cv2.imread(path)               # BGR, as YOLOX expects
        if img is None:
            return []
        tensor, ratio = preproc(img, self.test_size)   # 114-letterbox, no norm
        tensor = self.torch.from_numpy(tensor).unsqueeze(0).float().to(self.device)
        with self.torch.no_grad():
            out = self.model(tensor)                    # head decodes in eval mode
            # per-class NMS: a glasses box must not suppress an overlapping
            # rayban box (they cover the same object by design)
            out = postprocess(out, self.num_classes, conf, self.nmsthre,
                              class_agnostic=False)[0]
        boxes = []
        if out is not None:
            for row in out.cpu().numpy():
                x1, y1, x2, y2, obj, cls_conf, cls = row[:7]
                boxes.append((float(x1 / ratio), float(y1 / ratio),
                              float(x2 / ratio), float(y2 / ratio),
                              float(obj * cls_conf), int(cls)))
        return boxes


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_yolox_ckpt(weights):
    # YOLOX checkpoints are .pth living under a "yolox"-named run dir; RF-DETR
    # .pth checkpoints don't match this.
    if not weights.endswith(".pth"):
        return False
    tail = (os.path.basename(os.path.dirname(weights)) + " "
            + os.path.basename(weights)).lower()
    return "yolox" in tail


def build_detector(weights, device):
    # YOLO ships a .pt file; YOLOX a .pth under a yolox run dir; RF-DETR a dir/.pth.
    if weights.endswith(".pt"):
        return YoloDetector(weights, device)
    if _is_yolox_ckpt(weights):
        return YoloxDetector(weights, device)
    try:
        return RFDetrDetector(weights, device)
    except Exception:                      # noqa: BLE001 - fall back to YOLO loader
        return YoloDetector(weights, device)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="image file or folder")
    ap.add_argument("--weights", default=_DEFAULT_YOLO)
    ap.add_argument("--conf", type=float, default=0.4,
                    help="min rayban box confidence to call RAY-BAN META")
    ap.add_argument("--conf-glasses", type=float, default=0.35,
                    help="min glasses box confidence to call NORMAL GLASSES "
                         "(below both thresholds -> NO GLASSES)")
    ap.add_argument("--save", action="store_true",
                    help="write annotated images to detected_image_draw/")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    if not os.path.exists(args.weights):
        raise SystemExit(f"weights not found: {args.weights}\n"
                         "train first: python train_detector.py --model yolo")
    device = args.device or pick_device()
    det = build_detector(args.weights, device)

    floor = min(args.conf, args.conf_glasses)
    n_meta = n_normal = n_none = n_total = 0
    for path in _iter_images(args.target):
        n_total += 1
        boxes = det.detect(path, floor)
        boxes.sort(key=lambda b: -b[4])
        rb = max((b[4] for b in boxes if b[5] == 0), default=0.0)
        gl = max((b[4] for b in boxes if b[5] == 1), default=0.0)
        if rb >= args.conf:
            print(f"RAY-BAN META ({rb:.2f})     {path}")
            n_meta += 1
        elif gl >= args.conf_glasses:
            print(f"NORMAL GLASSES ({gl:.2f})   {path}")
            n_normal += 1
        else:
            print(f"NO GLASSES          {path}")
            n_none += 1
        if args.save:
            img = cv2.imread(path)
            if img is not None:
                out_path = os.path.join(OUT_DIR, os.path.basename(path))
                _annotate(img, boxes, out_path)
    if n_total > 1:
        print(f"\n{n_meta}/{n_total} RAY-BAN META, {n_normal} NORMAL GLASSES, "
              f"{n_none} NO GLASSES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
