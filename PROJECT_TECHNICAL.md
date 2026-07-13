# MODULE SCAN — Technical Design Document

**Project:** Ray-Ban Meta smart-glasses camera-module detector (`try3-non-ml`)
**Goal:** Given a photo (or live webcam frame) of glasses, decide whether they are **Ray-Ban Meta / Ray-Ban Stories smart glasses** (which carry a small camera module in a top-outer frame corner) versus **ordinary eyeglasses**.
**Runtime:** 100% in the browser — static HTML + OpenCV.js (WASM) + onnxruntime-web. No server, nothing uploaded.
**Deployment:** Static site on Vercel (`framework: null`; only `index.html` + `static/` are served).

---

## 0. Table of contents

1. Problem statement and design philosophy
2. System architecture (two detector generations)
3. The classical CV localization stage
4. The learned classification stage (the CNNs)
5. Training methodology
6. Threshold and tier derivation (the core of the design)
7. Datasets
8. The browser application (end-to-end runtime flow)
9. Results and accuracy
10. Model export & deployment
11. File-by-file reference
12. Known limitations & evolution

---

## 1. Problem statement and design philosophy

A Ray-Ban Meta looks almost exactly like a normal pair of glasses. The single distinguishing visual signal is a **small, dark, glassy, near-circular camera module set in a raised metallic bezel, with a tiny bright specular glint, located at the top-outer corner (end piece) of the frame** — and there is one in **both** corners.

The central difficulty: **a camera lens and an ordinary rounded frame corner are both "a dark circle in the corner."** No hand-tuned geometric rule cleanly separates them — the difference is in fine shape and texture. This is why the project uses machine learning for exactly *one* decision (is this corner a camera module?), while everything else — finding the face, the glasses, the corners — is done with classical, explainable computer vision.

Design principles:

- **Scale/structure robustness, not magic constants.** All spatial parameters are relative to frame width or the detected face/lens box, so behavior is invariant to image resolution after a canonical resize.
- **Both corners must fire.** A Ray-Ban has a module in *both* top-outer corners; a normal frame, a stray shadow, or a hand near one side cannot satisfy both. This is the strongest false-positive suppressor in the system.
- **Abstain rather than guess.** If the pipeline cannot isolate a glasses-shaped object (e.g. a bare selfie of a face), it returns NORMAL instead of guessing — because a human eye also reads as "a dark circle with a glint."
- **Zero *confident* false positives by construction.** The YES threshold is derived so that no normal-glasses image in the validation set can ever get two corners above it (see §6).
- **The privacy LED is never used** as a signal (it is not always on and is trivially defeated).

---

## 2. System architecture (two detector generations)

The repository contains **two parallel classifier systems** that share the localization code but are otherwise distinct. Understanding this split is essential.

| | **Corner-module CNN** (PRIMARY, shipped) | **Region CNN** (fallback) | **HOG + logistic** (legacy last-resort) |
|---|---|---|---|
| What it classifies | Each top-outer **corner** crop, at full resolution | The **whole glasses region** crop | Each corner crop (32×32) |
| Model | MobileNetV3-small, input 112 | MobileNetV3-small, input 160 | 144-dim HOG → logistic regression |
| Output | Per-corner P(module); 3-tier YES/MAYBE/NO | Single P(Ray-Ban); 2-tier META/NORMAL | Per-corner P(camera); 2-tier META/NORMAL |
| Weights | `models/corner_clf.pt` → `static/models/corner_clf.onnx` | `models/region_clf.pt` → `static/models/region_clf.onnx` | `models/camera_clf.npz` / `static/camera_clf.json` |
| Trained by | `train_corner_clf.py` | `train_region_clf.py` | `pipeline/camera_clf.py:train_logreg` |
| Localization | `pipeline/corners.py` | `pipeline/region.py` | `pipeline/locate.py` |

**Fallback chain at runtime (decided per frame):** Corner CNN → Region CNN → HOG. The corner CNN is tried first; if no face is found in the frame it returns `null` and the app falls through to the region CNN, then to HOG if neither ONNX model loaded.

**Shared pipeline skeleton:** `preprocess → (face detect) → locate/crop → classify → decide → visualize`. The Python CLI (`detect_meta_glasses.py`) wires the *HOG* path; the browser and the training scripts drive the *CNN* paths.

---

## 3. The classical CV localization stage

### 3.1 Preprocessing (`pipeline/*` / `static/detector/preprocess.js`)
- Canonical resize to **`target_width = 1000`** px (all downstream params are relative to this).
- Grayscale + **CLAHE** contrast equalization (`clahe_clip = 2.0`, `clahe_tile = 8` → 8×8 grid).
- Gaussian blur (`gauss_ksize = 5`, forced odd) for the segmentation gray.

### 3.2 Face detection (`pipeline/facedet.py` / `facedet.js`)
Pure Viola–Jones Haar cascades (no NN). Operating point tuned for **zero false faces on studio normal shots** while still catching worn faces:
- `ROTATIONS = (0,)` — upright only (rotated passes hallucinated faces).
- Cascades: `frontalface_default`, `frontalface_alt2`, and `profileface` (profile run only if frontals find nothing, on both the image and its horizontal mirror).
- `SCALE_FACTOR = 1.05`, `MIN_NEIGHBORS = 6`, `MIN_SIZE_FRAC = 0.10` (min face side = 10% of min(H,W)).
- Returns the **largest-area** box, or `null`.

### 3.3 Eye line (`pipeline/region.py:_eye_line` / `region.js:eyeLine`)
Within the face's upper 62%, runs `haarcascade_eye_tree_eyeglasses.xml` then `haarcascade_eye.xml` (`detectMultiScale(1.1, 4, minSize=0.12·face_w)`). Returns the **median eye-center y**; fallback `face.y + 0.38·face.h`. This vertical anchor positions the corner crops.

### 3.4 Corner cropping (`pipeline/corners.py` / `corners.js`) — the PRIMARY path
For each side (L/R), a square box is cut from the **original full-resolution image** (the module is only resolvable there — a 160px face crop loses it):
- Box side `w = corner_size · face.w` (`corner_size = 0.42`).
- Vertical center `cy = eye_cy − corner_yc · face.h` (`corner_yc = 0.05`).
- Horizontal: L → `face.x − corner_out · face.w`; R → `face.x + face.w + corner_out · face.w − w` (`corner_out = 0.06`).
- The **right corner is horizontally mirrored** to the left's canonical orientation, so both sides share training examples and the model is flip-symmetric.
- Resized to `corner_input = 112`, BGR→RGB.

A **candidate grid** of 18 boxes per side (`corner_grid_sizes = (0.36, 0.42, 0.52)` × `corner_grid_yc = (−0.02, 0.05, 0.14)` × `corner_grid_out = (0.0, 0.12)`) is used at evaluation time to absorb face-anchor imprecision (grid-max scoring, §6). The **center box** (middle of each axis) is the canonical training crop.

### 3.5 Region / lens localization (fallback paths)
- **Region crop** (`region.py:locate_region`): worn → box around the face width ± `region_wpad = 0.06`, vertical band `eye_cy − region_up·face.h` (`region_up = 0.24`) to `eye_cy + region_down·face.h` (`region_down = 0.20`); held → padded segmentation bbox; else center crop. Feeds the region CNN.
- **Lens-hole localization** (`locate.py`, HOG path): Otsu threshold + morphology inside the frame bbox, interior contours (`RETR_CCOMP`) ≥ 2% area = lens holes; largest per half is the lens. Corner ROIs are carved on the end piece outside each lens. A **face-search grid** (36 ROIs/side) is used for worn glasses.

### 3.6 Segmentation (`pipeline/segment.py`)
For held/studio shots without a face: Otsu/Canny → morphology → largest contour whose area (`seg_min_area_frac = 0.10` … `seg_max_area_frac = 0.97`) and aspect (`seg_min_aspect = 1.4` … `seg_max_aspect = 4.5`) match a glasses frame. Produces the bbox that gates the held path.

### 3.7 Domain gate (`locate.py:two_lens_gate` / `decide.js`)
Deliberately does **not** gate on precise lens geometry (that was over-rejecting real Ray-Bans). Face path passes iff a face was found; held path passes iff both lenses were located. If the gate fails → verdict NORMAL, "no glasses frame found."

---

## 4. The learned classification stage

### 4.1 Architecture (both CNNs)
- **Backbone:** `torchvision.models.mobilenet_v3_small(weights=IMAGENET1K_V1)`.
- **Head:** `classifier[3]` replaced with `nn.Linear(in_features, 1)` — a single logit.
- **Output:** `sigmoid(logit)` → probability. The browser applies the sigmoid; the ONNX graph exports the raw `logit`.
- **Normalization:** ImageNet `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`.
- **Corner variant:** input 112, classifies one corner crop. Runs a **batch of 2** in the browser (both corners in one ORT call).
- **Region variant:** input 160, classifies the whole region crop (batch 1).

### 4.2 Legacy HOG classifier (`pipeline/camera_clf.py`)
- 32×32 gray crop → per-crop illumination normalize → `cv2.HOGDescriptor((32,32),(16,16),(16,16),(8,8),9)` = **144-dim** descriptor → standardize `(x−μ)/σ` → logistic regression `sigmoid(x·w + b)`.
- Trained by full-batch gradient descent (`train_logreg`, L2 = 1.0, lr = 0.1, 2000 iters).
- Because OpenCV.js does not expose `HOGDescriptor`, the browser uses a **re-implemented 144-dim HOG** (`hog.js`: 8×8 cells, 9 unsigned bins, 2×2 blocks, L2-Hys clip 0.2) with a classifier **retrained on that exact descriptor** — the `.npz` and the browser JSON are deliberately *not* synced.

---

## 5. Training methodology (`train_corner_clf.py`)

**Hyperparameters:** `SEED = 0`, `INPUT = 112`, `BATCH = 32`, `EPOCHS_HEAD = 12`, `EPOCHS_FT = 8`, `KFOLDS = 4`. Device: mps > cuda > cpu.

**Two-stage transfer learning (`train_one`):**
1. Freeze backbone, train head only — Adam(lr = 1e-3), 12 epochs.
2. Unfreeze all, fine-tune — Adam(lr = 1e-4, weight_decay = 1e-4), 8 epochs.
- **Loss:** `BCEWithLogitsLoss` (no `pos_weight`; class balance handled by the sampler).

**Class balancing:** `WeightedRandomSampler` over three tiers with target mass `{pos: 0.5, indomain: 0.4, bulk (MeGlass): 0.1}`, renormalized over present tiers.

**Augmentation (corner):**
- `ToImage → Resize(112·1.15) → RandomResizedCrop(112, scale=(0.7,1.0), ratio=(0.85,1.18))`
- **`DownscaleJitter(48→112, p=0.3)`** — randomly downscales then upscales, so *sharpness is not a class shortcut* (MeGlass negatives are ~120px blurry, positives are sharp; without this the model would learn "sharp = Ray-Ban").
- `ColorJitter(0.3,0.3,0.3,0.05) → RandomRotation(8) → Normalize`.
- **No horizontal flip** (corners are already mirrored to canonical orientation).

*(Region CNN differs: input 160, `RandomResizedCrop(scale=(0.65,1.0))`, `RandomHorizontalFlip()` **is** used, `RandomRotation(10)`, no DownscaleJitter.)*

**Cross-validation:** grouped **4-fold** at the image level. Images are grouped by identity/product (`_group_key`) so near-duplicates never straddle train/test. Folds are assigned round-robin per class for balance. Out-of-fold (OOF) predictions drive all threshold selection; the **final shipped model is retrained on all data**.

**Grid aggregation:** each corner is scored across its 18-box grid; `AGGREGATE = "gridmax"` (default) takes the max box; the `"second_high"` alternative requires two overlapping boxes to agree (a stronger FP suppressor). The center box is excluded from the statistic.

**Dedup & exclusions:** exact-duplicate files removed by md5; 14 positive corner *sides* excluded from training (via `models/corner_exclusions.json`) because the module is invisible at extreme angle/occlusion — these would be label noise. Positives' *off-center* grid crops are eval-only (`train_use = False`) for the same reason; negatives contribute **all** grid positions as hard negatives.

**Training scale (from `models/corner_train.log`):** 5808 corner crops from 564 images (2318 Ray-Ban corners, 3490 normal corners); 14 excluded sides; 4 dedup duplicates removed.

---

## 6. Threshold and tier derivation — the core of the design

This is where the "zero confident false positives" guarantee comes from. Everything is computed on **OOF predictions**, at the **image level**, on the **worn (real) subset**.

### 6.1 Single FP-first threshold (`pick_threshold`)
- **Floor A** = `nextafter(max over worn in-domain negatives of their score, 1)` → clears *every* worn normal-glasses image (zero worn FP by construction).
- **Floor B** = `quantile(MeGlass bulk negatives, 1 − fp_budget_bulk)` = 0.99 quantile (allows ≤1% bulk FP).
- `q_neg = max(A, B)`; `q_pos = quantile(positives, 1 − target_recall)` = 0.10 quantile.
- Threshold = midpoint of the gap if `q_pos > q_neg`, else `q_neg`. Clamped to `[0.05, 0.999]`.

### 6.2 Three-tier thresholds (`pick_tier_thresholds`) — the shipped rule
- **`tc_hi`** = `nextafter(max over normals of their *second-highest* corner score, 1)`, capped 0.999. **By construction, no normal image can have two corners ≥ `tc_hi`** → **zero confident-YES false positives.**
- **`tc_maybe`** = smallest threshold at which the count of normals with a max-corner ≥ it stays within a review budget (default 10%; retunable to 25% via `retune_tier_maybe.py`), clamped `[0.5, tc_hi]`.

### 6.3 The verdict (`tier_verdict` / `corners.js:tier`)
```
hits = (pL ≥ tc_hi) + (pR ≥ tc_hi)
hits ≥ 2                        → YES   (both corners = smart glasses)
pL ≥ tc_maybe OR pR ≥ tc_maybe  → MAYBE (one corner — check manually)
otherwise                       → NO    (normal glasses)
```

**Shipped thresholds (`corner_clf.meta.json`):** `tc_hi = 0.9519512653350831`, `tc_maybe = 0.7861909547738695`, input 112, ImageNet mean/std, `corner_size = 0.42`, `corner_yc = 0.05`, `corner_out = 0.06`.

*(An earlier baseline used `tc_hi = tc_maybe = 0.936`; the region CNN uses a single `threshold = 0.275`.)*

---

## 7. Datasets

| Directory | Count | Role |
|---|---|---|
| `ray_ban_face/` | 62 (58 after dedup/exclusions) | **Positive** — Ray-Ban Meta worn on faces |
| `ray_ban_frame/` | 17 | **Positive** — clean studio Meta frames |
| `actual_rayban_cases/` | 2 | **Positive** — real straight-on Meta captures |
| `normal_glassess/` | 69 (41 "in-domain" subset) | **Negative** — ordinary worn/held glasses |
| `normal_frame/` | 25 | **Negative** — clean studio normal frames |
| `data/normal/meglass/` | 500 (434 used) | **Negative bulk** — MeGlass public dataset of ordinary eyeglasses |
| `data/rayban/`, `data/normal/` | drop-in | user-added extras |

**Polarity:** positive = Ray-Ban Meta / Stories (camera module in corner); negative = normal eyeglasses.

**MeGlass import (`tools/import_meglass.py`):** selects the eyeglasses-ON subset, **face-gates** each image (keeps only those where region localization returns a real worn box), samples **one image per identity** (up to 500) for diversity, copies originals byte-for-byte with a manifest. Provides scale to the hard-negative set.

---

## 8. The browser application (end-to-end runtime)

### 8.1 Loading (`static/app.js:init`)
On `DOMContentLoaded`, four parallel loads: corner CNN (`loadCornerModel`), region CNN (`loadCnnModel`), HOG JSON, and OpenCV.js (`whenCvReady` polls `cv.Mat`) → Haar cascades (`loadFaceCascades`: 3 face + 2 eye XMLs). ONNX runs via onnxruntime-web v1.20.1, `wasm` provider, `numThreads = 1` (avoids cross-origin isolation), `wasmPaths` set to an absolute `static/ort/` URL. Readiness = `cvReady && facesReady && (cornersReady || cnnReady || model)`.

### 8.2 Guided capture flow
Live webcam with a three-pose sequence (`STEPS`): **LEFT ◀**, **RIGHT ▶** (captured but not analyzed), then **FRONT · STRAIGHT ON** (analyzed). Constants: `CYCLE_SECONDS = 5` countdown, `RESULT_HOLD_MS = 3000`, `BURST_FRAMES = 5`, `BURST_GAP_MS = 200`. On START, a continuous **screen recording** begins (`getDisplayMedia`, MediaRecorder → WebM). The straight pose grabs a **5-frame burst** over ~1 s.

### 8.3 Per-frame detection (`runDetect`)
Image → canvas → `cv.matFromImageData` → RGBA Mat. Fallback chain:
1. Corner CNN (`corners.detect`): gray → face detect → **no face ⇒ return `null`** (fall through); else build both corner boxes (R mirrored), one batched `[2,3,112,112]` ORT call, `sigmoid` per corner, `tier(pL, pR)`.
2. Region CNN (`detectRegion`): locate region → crop → `cnn.infer` → `prob ≥ 0.275`.
3. HOG (`detectMat`): grid-scored corner crops, `min` peak per side, both ≥ 0.50.

### 8.4 Multi-frame aggregation (`aggregateBurst`)
Corner path: **median per corner** across the 5 frames, then re-run `tier()`. Median = majority vote for odd counts, yields a calibrated meter value, and rejects ≤2 outlier frames (blur/blink/face-wobble). Region path: median probability.

### 8.5 Rendering (`renderResult`, `buildOverlay`)
- Verdict: YES→"SMART GLASSES" (red), MAYBE→"MAYBE — CHECK" (amber), NO→"NORMAL GLASSES" (green).
- Confidence meter: `min(L,R)` when YES (the weaker corner the verdict hinges on), `max(L,R)` otherwise; region path shows the single prob. A marker sits at the threshold.
- Per-corner cards (CAMERA / camera? / no camera), a plain-English "why" explanation, and an annotated overlay canvas (face box, corner ROIs with `Pcam=` labels, colored banner).
- END downloads all captured pose PNGs + the WebM recording.

### 8.6 Deployment
`vercel.json`: `framework/buildCommand/installCommand = null`, `outputDirectory = "."` — Vercel just serves files. `.vercelignore` strips all Python, models, datasets, and the README so it is not mis-detected as a Flask app. Must be served over http(s) (not `file://`).

---

## 9. Results and accuracy

### 9.1 File-by-file (shipped tiers, `models/corner_accuracy.txt`)

| Dataset | Goal | Count | Correct | Accuracy |
|---|---|---|---|---|
| Ray-Ban worn (flag = YES/MAYBE) | flag | 58 | 43 | **74%** |
| Normal in-domain (want NO) | NO | 41 | 24 | **59%** |
| Normal MeGlass (want NO) | NO | 434 | 434 | **100%** |

- Ray-Ban breakdown: YES = 25, MAYBE = 18, NO = 15 (misses).
- Normal in-domain: YES = 0, MAYBE = 17 (soft FP), NO = 24.
- **Confident false positives (a YES on any normal image): 0 / 475 = 0.00%.**

### 9.2 Out-of-fold cross-validation

| Report | Model | Acc | Recall | FP-rate | Notes |
|---|---|---|---|---|---|
| `corner_oof_report.json` | Corner (shipped, tc_hi 0.952) | 90.1% | 16.4% @ thr 0.999 (93.1% @ 0.25) | 0.99% | FP-first, tight tiers |
| `corner_oof_report.baseline.json` | Corner (tc 0.936) | 92.0% | 32.8% | 0.42% | matches accuracy.txt tiers |
| `oof_report.json` | Region CNN | 95.4% | 41.7% (studio 82.4%) | 0.0% | best overall acc |

**Trade-off:** the project deliberately trades recall for a near-zero false-positive rate. Recall on worn Ray-Bans is high at a low threshold (93% @ 0.25) but the shipped tiers are tuned FP-first so that no normal-glasses image is ever confidently flagged.

---

## 10. Model export & deployment (`tools/export_corner_onnx.py`)

- Load `corner_clf.pt` → build meta (`input, mean, std, tc_hi, tc_maybe, corner geometry`).
- `torch.onnx.export` opset 17, dynamic batch axis, `dynamo=False`.
- **Static int8 quantization** (`quantize_static`, QDQ, per-channel, QInt8) with ≤200 stride-sampled calibration crops.
- **Validation:** count per-corner tier-flips (at tc_hi/tc_maybe) between int8 and fp32; if int8 flips > 5% of samples, **ship fp32 instead**.
- Writes `static/models/corner_clf.onnx` + `corner_clf.meta.json`. (Region analogue `export_onnx.py` ships fp32 if verdict agreement < 95%.)

Supporting tools: `make_corner_exclusions.py` (rebuild the exclusion list from the OOF report), `retune_tier_maybe.py` (re-derive `tc_maybe` without retraining), `analyze_corner_agg.py` (offline sweep of grid-aggregation statistics).

---

## 11. File-by-file reference

**Python pipeline:** `config.py` (all tunables), `pipeline/facedet.py` (Haar face), `pipeline/region.py` (region + eye line), `pipeline/corners.py` (corner cropping), `pipeline/locate.py` (lens/ROI, HOG path), `pipeline/segment.py` (frame bbox), `pipeline/features.py` + `pipeline/camera_clf.py` (HOG features + logreg), `pipeline/decide.py` (verdict), `pipeline/viz.py` (overlay + payload).
**Training:** `train_corner_clf.py`, `train_region_clf.py`, `datasets_corners.py`, `datasets.py`.
**Tools:** `export_corner_onnx.py`, `export_onnx.py`, `make_corner_exclusions.py`, `retune_tier_maybe.py`, `analyze_corner_agg.py`, `import_meglass.py`.
**CLI:** `detect_meta_glasses.py`.
**Browser:** `index.html`, `static/app.js`, `static/detector/{config,preprocess,segment,facedet,region,cnn,corners,locate,decide,index,hog,camera_clf}.js`, `static/style.css`, `static/models/*.onnx` + `*.meta.json`, `static/ort/`, `static/haar/`.

---

## 12. Known limitations & evolution

**Limitations:**
- Recall on worn glasses is modest (74% flagged, only 25/58 confident YES) — the design prioritizes precision.
- Corner CNN requires a detectable **face**; no face ⇒ falls back to the region CNN.
- Hardest cases: strongly off-angle / side-on Meta views (module invisible), and cat-eye normal frames whose pointed corner mimics a module (soft MAYBE false positives).
- MeGlass negatives are low-resolution (~120px); handled via DownscaleJitter but still a domain gap.

**Evolution (git + artifacts):**
1. Pure classical CV pipeline.
2. HOG + logistic-regression corner verifier (~19 studio images; the model described in `manager_explainer.html`, now out of date).
3. Full browser port via OpenCV.js + re-implemented JS-HOG; Flask removed.
4. Region CNN (MobileNetV3-small, ONNX) — best overall accuracy.
5. **Current: corner CNN** — scaled up with 500 MeGlass bulk negatives + worn positives, tuned FP-first to zero confident false positives, three-tier YES/MAYBE/NO. This is the shipped primary detector.

> Note: `README.md` and `manager_explainer.html` describe earlier generations (the HOG model, ~84% LOO-CV, 19 images) and are out of date relative to the corner-CNN artifacts documented here.
