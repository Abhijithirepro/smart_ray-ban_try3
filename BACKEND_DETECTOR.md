# Backend Ray-Ban Meta Detector — Status & Handoff

Everything below was built in this session. The **old** browser detector
(`static/`, `index.html`, `pipeline/`) is untouched — this is a parallel backend
object detector. Goal: detect Ray-Ban Meta glasses (one box around the whole
glasses, camera inside) and NOT fire on normal glasses.

---

## ⏳ Current status (as of this handoff)

- **A training run is in progress** in the background: pretrained `yolo11s.pt`,
  60 epochs, imgsz 960, on Apple MPS. It is **slow (~minutes/epoch at 960px)**.
- Weights land at `runs/rayban_yolo/weights/best.pt` when done.
- The earlier from-scratch run is preserved at `runs/rayban_yolo_scratch/`.
- Log: `runs_train.log` (also `runs/rayban_yolo/results.csv` for per-epoch metrics).

### When you come back — check if training finished
```bash
cd "<project root>"
tail -20 runs_train.log
cat runs/rayban_yolo/results.csv        # one row per finished epoch
ls -lh runs/rayban_yolo/weights/best.pt # exists once >=1 epoch saved
ps aux | grep "[t]rain_detector"        # empty = training finished (or was killed)
```

### If training was KILLED mid-run (background procs die on account switch / sleep)
Resume from the last checkpoint — keeps all completed epochs and the LR schedule:
```bash
nohup .venv-train/bin/python train_detector.py --model yolo --resume \
    > runs_train.log 2>&1 &
```
(The run is 60 epochs at ~6 min/epoch on MPS ≈ several hours. It has been killed and
resumed before; just re-run the above each time it dies until `results.csv` reaches
epoch 60 or early-stops.)

### Progress so far (epoch 12 checkpoint, before final run)
FP fix is already working at conf 0.35: studio Ray-Bans 29%→**100%** detected;
false positives on worn normal glasses **75%→32%**; **all 18 old-baseline FPs fixed**
at conf 0.45; val mAP50 **0.671** (2× the from-scratch 0.31). Recall is still held
back by compressed confidence scores — that is what the remaining epochs fix.

---

## The pipeline (all commands use `.venv-train/bin/python`)

| Step | File | What it does |
|---|---|---|
| 1. Bootstrap boxes | `tools/bootstrap_boxes.py` | Auto-draws one whole-glasses box per positive by reusing `pipeline/region.locate_region`. Saves **original-resolution** images + `data/boxes/boxes.json`. Negatives saved boxless (background). |
| 2. Review (optional) | `tools/label_review/serve.py` | Local web UI (localhost:8765) to drag/fix/tighten boxes → rewrites `boxes.json`. **Recommended: tighten the ~19 loose `held`/`center` boxes.** |
| 3. Augment | `tools/augment_positives.py` | Failure-mode augmentation. Positives ×8 (downscale/occlusion/lighting/flip). Negatives ×3 + glare (`--neg-per-image`, added to fix false positives). → `data/boxes/aug.json` |
| 4. Export | `tools/export_dataset.py` | Renders `boxes.json`+`aug.json` to `--format yolo` (`data/rayban_yolo/`) or `--format coco` (`data/rayban_coco/`, for RF-DETR). |
| 5. Train | `train_detector.py` | `--model yolo` (Ultralytics YOLO11/12) or `--model rfdetr` (needs GPU). |
| 6. Infer | `infer.py` | `python infer.py IMAGE --save` → verdict + annotated overlay to `detected_image_draw/`. |
| 7. Mark cameras | `mark_cameras.py` | Draws box + "camera" crosshairs at the top-outer corners (where Ray-Ban modules sit) → `detected_cameras/`. |
| 8. Evaluate | `eval_detector.py` | `--sweep` prints a conf→precision/recall table and auto-picks an FP-first threshold; writes `detection_accuracy_yolo.csv`; `--val-map` adds mAP. |
| Resize (standalone) | `tools/resize_images.py` | Non-destructive 640×480 letterbox copies → `data/resized_640x480/`. NOT used by training. |

---

## What we're fixing right now: normal glasses detected as Ray-Ban

Measured on the from-scratch model: **52/94 normal images false-fired (75% of worn
normals)**; Ray-Ban recall 70%. Root causes + fixes (this is the run in progress):

1. **No pretrained features** → now training from `yolo11s.pt` (COCO-pretrained).
   *(github is blocked in this env; downloaded from Hugging Face instead —
   `https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11s.pt`.)*
2. **Class imbalance we created** (612 pos vs 87 neg) → negative augmentation added
   (`--neg-per-image 3`): now ~612 pos vs ~1063 neg.
3. **Loose boxes** on 19 studio/fallback positives → fix in the review UI (step 2).
4. **Threshold too low** (0.25) → `eval_detector.py --sweep` picks an FP-first conf.

### After training finishes — run this to see the fix
```bash
.venv-train/bin/python eval_detector.py --sweep --val-map        # picks best conf, prints table
rm -rf detected_cameras
.venv-train/bin/python mark_cameras.py ray_ban_face ray_ban_frame actual_rayban_cases \
    normal_glassess normal_frame --conf <chosen-conf> --all
```
Success = FP≈0 on `normal_glassess` (e.g. `03-straight (5).png` clean) while Ray-Ban
recall stays high. Compare to old baseline in `detection_accuracy.csv` (TP=41 FP=18
TN=51 FN=17) and scratch model (TP=57 FP=52).

---

## Cheat-sheet: full rebuild from scratch
```bash
cd "<project root>"
P=.venv-train/bin/python
$P tools/bootstrap_boxes.py --max-meglass 200 --overlays
# (optional) $P tools/label_review/serve.py   → tighten boxes at localhost:8765
$P tools/augment_positives.py --per-image 8 --neg-per-image 3
$P tools/export_dataset.py --format yolo
$P train_detector.py --model yolo --weights yolo11s.pt --epochs 60 --imgsz 960 --patience 20
$P eval_detector.py --sweep --val-map
```

## ⚠️ Environment gotchas (important)
- **Use `.venv-train/bin/python -m pip`**, NOT `.venv-train/bin/pip` — the pip shim
  has a stale path (installs to the wrong site-packages; packages "install" but
  won't import).
- **opencv conflict:** if you see `cv2 has no attribute CascadeClassifier`, run
  `.venv-train/bin/python -m pip uninstall -y opencv-python-headless` (a 5.0 headless
  build shadows the working system opencv 4.13). albumentations still works after.
- **Network:** github.com is blocked in this environment; huggingface.co and pypi
  work. Pretrained weights come from HF (see above).

## To get even better accuracy (after this run)
1. **More real Ray-Ban photos** — drop into `data/rayban/` (already a source folder),
   then rerun the cheat-sheet. Biggest ceiling-raiser. Prioritize clear/light frames,
   profiles, and 1–2 m distance shots.
2. Tighten the 19 loose boxes (review UI).
3. Bigger model: `--weights yolo11m.pt`, or `--model rfdetr` on a GPU.
4. Higher res: `--imgsz 1280` (camera cue is tiny).

## File map (new this session)
```
tools/bootstrap_boxes.py      tools/augment_positives.py    tools/export_dataset.py
tools/label_review/           tools/resize_images.py
train_detector.py             infer.py    eval_detector.py    mark_cameras.py
yolo11s.pt (pretrained)       runs/rayban_yolo/ (training)   runs/rayban_yolo_scratch/
data/boxes/ (images+boxes.json+aug.json)   data/rayban_yolo/   data/rayban_coco/
detected_cameras/   detected_image_draw/   detection_accuracy_yolo.csv
```
Reused unchanged: `pipeline/region.py`, `pipeline/preprocess.py`, `config.py`,
`datasets.py`. Plan file: `~/.claude/plans/if-you-re-training-an-glistening-wave.md`.
