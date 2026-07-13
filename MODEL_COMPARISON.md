# Detector model comparison — YOLO11s vs YOLOX-Nano

Goal: replace the backend Ray-Ban Meta detector (YOLO11s, ~9.4M params, 18 MB) with a
much smaller model that can eventually run in the browser. This documents a like-for-like
comparison so the size/accuracy trade-off is explicit.

Single-class detector (`rayban_meta`); normal / no-glasses images are background
negatives. All three rows are evaluated with the **same harness** (`eval_detector.py`),
whose folder sweep over the in-domain folders is the primary decision metric (the COCO/
Ultralytics val split is tiny — 13 positives — so val mAP is noisy).

## Environment
- macOS, Apple Silicon (MPS), `.venv-train` (torch 2.12, numpy 2.4, Python 3.14).
- YOLOX: Megvii repo pinned at commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee`,
  vendored under `third_party/YOLOX` with `tools/yolox_mac.patch` applied
  (CUDA → MPS/CPU device handling; `.type('torch.*.FloatTensor')` string casts
  replaced with device-agnostic `.to()` / `.type_as()`).
- Training data (both models train on the exported box manifest):
  positives `ray_ban_face` + `ray_ban_frame` + `actual_rayban_cases` (81 boxed);
  negatives `normal_glassess` + `normal_frame` + MeGlass (capped 200) +
  **`hirepro_images_normal/8-july` (200, NEW)**. After offline augmentation:
  **train 2404 images / 612 boxes, val 94 / 13**.

## Verdict

**YOLOX-Nano wins decisively — ~10× smaller AND substantially more accurate.** At the
FP-safe operating point (≤1 false positive on normals), it reaches **recall 0.938 /
precision 0.987**, where the old YOLO11s collapsed to **recall 0.025** and the old CV
pipeline sat at recall 0.707. It also recovers **15/17** of the old system's misses and
fixes **18/18** of its false alarms.

## Results

| model | params | GFLOPs | ckpt size | onnx size | imgsz | val mAP50 / mAP50-95 | folder sweep P / R @conf0.30 | FP≤1 operating point | train time / hw |
|---|---|---|---|---|---|---|---|---|---|
| old CV pipeline | — | — | — | — | — | — | P 0.695 / R 0.707 (whole run) | — | — |
| **YOLO11s** (baseline) | 9.41M | 21.3 | 18 MB (`.pt`) | ~36 MB fp32 | 640 | 0.664 / 0.273 | 0.600 / 0.815 | conf 0.45 → **R 0.025** | MPS (finetune) |
| **YOLOX-Nano** | 0.90M | 2.55 | 7.5 MB (`.pth`) | **1.1 MB** fp32 | 640 | **0.768 / 0.488** | **0.974 / 0.938** | conf 0.40 → **R 0.938 / P 0.987** | ~3.5 h / MPS |

**Folder-sweep detail (YOLOX-Nano, `detection_accuracy_yolox.csv`, 175 in-domain images):**
at the recommended conf 0.40 → TP=76 FP=1 TN=93 FN=5, **precision 0.987, recall 0.938,
accuracy 0.966**. Failure-mode regression vs the old CV run: old false-negatives recovered
**15/17**, old false-positives fixed **18/18**.

**HirePro domain check (`hirepro_images_normal/8-july`, 200 normal/no-glasses images):**
**0 false positives on all 200** at conf 0.40. Split honestly — 169 were in the training
set, but the **31 held-out images (never trained on) also scored 0 FP / 100%**, showing the
zero-false-alarm behaviour generalizes to unseen HirePro-domain photos rather than
memorizing. (This folder is excluded from the main folder sweep above precisely because
most of it is training data; reported separately here.)

**Size wins:** 0.90M vs 9.41M params (**~10×**), 2.55 vs 21.3 GFLOPs (**~8×**), 7.5 MB vs
18 MB checkpoint, and a **1.1 MB** fp32 ONNX for the browser (verified bit-identical to the
torch model on positive + negative test images). ONNX I/O: input `images [1,3,640,640]`,
output `output [1,8400,6]` (raw grid outputs; decode at runtime per
`third_party/YOLOX/yolox/utils/demo_postprocess` — needed for the later frontend port).

Artifacts: weights `runs/rayban_yolox/best_ckpt.pth`, ONNX
`runs/rayban_yolox/yolox_nano_rayban.onnx`, per-image results `detection_accuracy_yolox.csv`,
COCO val log `yolox_coco_eval.log`, training log `yolox_train.log`.

## Baseline sources
- YOLO11s params/GFLOPs and val: `eval_out.log` ("YOLO11s summary (fused): 9,413,187
  parameters, 21.3 GFLOPs"), `yolo_finetune2.log` (best.pt val P=1.0 R=0.488
  mAP50=0.664 mAP50-95=0.298), `runs/rayban_yolo/results.csv`.
- YOLO11s folder sweep + FP-first point: `eval_out.log` / `detection_accuracy_yolo.csv`.
- Old CV pipeline: `detection_accuracy.csv` (TP=41 FP=18 TN=51 FN=17 → R 0.707, P 0.695).

## Caveats (read before drawing conclusions)
1. **Dataset differs from the recorded YOLO11s run.** YOLOX-Nano trains on ~200 extra
   `hirepro_images_normal/8-july` negatives that were NOT in the recorded YOLO11s dataset
   (verified: 0 hirepro records in the previous `data/boxes/boxes.json`). For a strictly
   fair head-to-head, retrain YOLO11s on the current dataset
   (`python train_detector.py --model yolo`, ~1–2 h) and add a row here.
2. **AP protocols differ.** YOLOX reports COCO-protocol AP (pycocotools); the YOLO11s
   baseline used the Ultralytics validator. Close but not identical — the folder sweep is
   the apples-to-apples number.
3. **Tiny val set** (13 positives): best-checkpoint selection by val AP is noisy. Prefer the
   folder-sweep operating point for decisions.
4. Model size is the point of this migration: **0.90M vs 9.41M params (~10×), 2.55 vs
   21.3 GFLOPs (~8×), ~7.5 MB vs 18 MB checkpoint.** Accept some recall loss if the
   FP-safe operating point holds.

## Escalation ladder (if YOLOX-Nano recall at the FP≤1 point is unacceptable)
1. Longer schedule / stronger mosaic (`mosaic_prob 1.0`, larger `no_aug_epochs`).
2. **YOLOX-Tiny** (5.06M params, width 0.375, non-depthwise) — still ~4× smaller runtime
   than YOLO11s; swap `depth/width` + `depthwise=False` in `exps/rayban_yolox_nano.py`.
