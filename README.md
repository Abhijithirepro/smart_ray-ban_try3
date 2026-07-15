# MODULE SCAN — Meta smart-glasses camera detector

A small, transparent tool that decides whether a clean, roughly front-on photo
of a pair of glasses is a pair of **Ray-Ban Meta / Ray-Ban Stories** smart
glasses (which carry a small camera module in a top-outer corner of the frame)
versus **normal eyeglasses**.

The pipeline locates the glasses with classical CV — Haar face/eye cascades for
worn glasses, thresholding/contours for held or studio shots — and crops the
whole glasses **region**. A **learned CNN classifier** (MobileNetV3-small,
transfer-learned, exported to ONNX and run in-browser via **onnxruntime-web**)
then calls Ray-Ban-vs-normal on that region. See `train_region_clf.py`,
`pipeline/region.py`, and `static/models/region_clf.onnx`.

> **Why the CNN?** The original detector used a HOG + logistic classifier on each
> top-outer corner crop (`models/camera_clf.npz`, still present as a fallback).
> It is excellent on clean studio shots but provably **cannot** separate a
> Ray-Ban camera module from an ordinary frame corner in real *worn* photos (both
> reduce to "a dark blob by a face") — it flagged ~90% of people in normal glasses
> as smart glasses. The whole-region CNN learns that distinction from data;
> false positives on real worn normal glasses drop from ~90% toward the
> single digits and keep improving as more training data is added (see `data/`).

---

## The signal it looks for

On a Ray-Ban Meta frame the camera appears as a **small, dark, glassy,
near-circular module** set in a slightly raised metallic **bezel**, usually with
a tiny bright **specular glint** at its centre, in the **top-outer corner / end
piece** where the top rim meets the temple hinge.

The detector scans **both** top-outer corners and fires META iff **both** read
as a camera (Ray-Ban frames carry a module in each corner). The right corner is
mirrored to a canonical orientation before classification, so left/right share
training examples and the result is robust to photo flip. The privacy LED is
**never** used as a signal.

The classifier learns the camera's appearance from corner crops rather than
checking a fixed checklist of cues — the HOG descriptor encodes the circular
bezel of a real module versus the L-shaped edge of a bare frame corner, which is
exactly the distinction the old hand-tuned circle/glint/darkness cues could not
make.

---

## Install

```bash
cd try3-non-ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt    # numpy, Pillow, opencv-python-headless
# (named requirements-dev.txt, not requirements.txt, so Vercel does not
#  auto-detect this static site as a Python backend)
# for the web UI also:
pip install flask
```

`opencv-python-headless` is fine for both the CLI and the web app (no GUI used).

---

## Use it — command line

```bash
python detect_meta_glasses.py IMAGE [options]
```

```
META  Pcam L=1.00 R=1.00  (ray_ban_frame/0rw4013__6702ce__p21__shad__fr.png)
```

Options:

| flag | meaning |
|------|---------|
| `--debug` | write an annotated overlay + JSON to `--debug-dir` (default `debug/`) |
| `--debug-dir DIR` | where debug artefacts go |
| `--json` | print the full result dict (verdict, per-corner `cam_prob`, geometry) |
| `--config FILE` | JSON file of config overrides (see `config.py`) |
| `--target-width N` | canonical resize width (default 1000) |
| `--threshold T` | override `cam_clf_thresh`, the per-corner P(camera) needed to fire |
| `--bbox x,y,w,h` | manual frame bounding box (skip segmentation) |
| `--quiet` | print just the verdict word |

The verdict is the payload; exit code is `0` unless the image can't be read.

## Use it — 100% in the browser (no server)

The whole pipeline runs **entirely in the browser** — no Python, nothing
uploaded. It just needs to be *served* over http(s) (the camera and the
OpenCV.js WASM need a secure context; opening the file directly with `file://`
won't work):

```bash
python -m http.server 8000   # then open http://localhost:8000
```

Drag a photo (or click) onto the scanner, or use live camera. It shows the
**verdict**, a **confidence meter** with the decision threshold marked, the
**per-corner P(camera)** from the classifier, and the **annotated overlay**
(frame bbox, lens boxes, the two corner ROIs labelled with their probability).

It works the same offline-capable way on any static host (GitHub Pages, Netlify,
etc.). How it maps to the Python pipeline:

- **`static/opencv.js`** — vendored OpenCV.js (WASM embedded; self-contained).
- **`static/detector/*.js`** — JS ports of `preprocess → segment → locate →
  corner-crop → HOG → logistic → decide`.
- **`static/camera_clf.json`** — the classifier weights. `cv2.HOGDescriptor` is
  not exposed by OpenCV.js, so the browser uses its own deterministic HOG
  (`static/detector/hog.js`) and the model is **retrained in that feature space**
  (`tools/train_js_clf.mjs`) for train/inference consistency.

### Parity vs the Python detector

Validated against the bundled reference set:

- **Verdict agreement (full JS pipeline, identical decoded pixels): 42/42** —
  `node tools/parity_node.mjs` (after `python tools/dump_crops.py && python
  tools/dump_raw.py`).
- **Leave-one-image-out CV (JS-HOG model): 90%** — `node tools/train_js_clf.mjs`.
- **Real-browser agreement: 41/42** — `browser_test.html`. The one miss is a
  hard-shadowed RGBA Ray-Ban render where the browser's PNG decoder yields
  slightly different pixels than `cv2.imread` (a near-threshold left corner). The
  live-camera path is unaffected (webcam frames are opaque, no PNG decode).

---

## How it works (pipeline)

```
preprocess → segment → locate → features → decide → (viz)
                                    │
                            camera_clf (per-corner P(camera))
```

| module | does |
|--------|------|
| `pipeline/preprocess.py` | load (OpenCV→Pillow fallback, flattens transparent PNGs onto white), canonical resize, grayscale + CLAHE + blur variants |
| `pipeline/segment.py` | frame bbox + filled mask via inverse-Otsu → morphology → largest contour, with a Canny fallback and a sanity gate; flags whether an **isolated** frame was found |
| `pipeline/locate.py` | derive a **generous top-outer search region** per side, relative to the frame bbox (scales with the frame, so near/far is handled) |
| `pipeline/features.py` | take each corner ROI's orientation-canonical crop and attach the learned **`cam_prob`** from `camera_clf` — no hand-tuned cues |
| `pipeline/camera_clf.py` | the classifier: corner crop → 32×32 (R mirrored to canonical) → illumination-normalise → HOG → standardise → logistic regression → `P(camera)` |
| `pipeline/decide.py` | fire META iff **both** corners are cameras (`cam_prob ≥ cam_clf_thresh`). **Domain gate**: abstain (NORMAL) if no isolated glasses frame was found |
| `pipeline/viz.py` | annotated overlay (frame bbox, lens boxes, the two corner ROIs labelled with per-corner `Pcam`) + JSON dump |

Retrain the verifier with `python3 train_camera_clf.py` (positives
`ray_ban_frame/`, negatives `normal_frame/`); it prints leave-one-image-out CV
and rewrites `models/camera_clf.npz`.

All spatial parameters are **relative** to the detected frame (bbox width /
height), not absolute pixels, so the detector is scale-invariant after the
canonical resize.

### The domain gate

The detector is built for the stated operating condition: **a clean photo of
just the glasses**. If segmentation can't isolate a glasses-shaped frame (e.g. a
photo of a *person wearing* glasses, where the frame fills the image or
segmentation collapses), it returns `NORMAL` with a reason rather than guessing —
because in a face photo an **eye** (a dark circle with a glint) can mimic a
camera module.

---

## Current accuracy (on the bundled reference sets)

Evaluate with:

```bash
python eval_separation.py
```

| set | result |
|-----|--------|
| `ray_ban_frame/` (clean Meta frames, positives) | **17 / 17 META** |
| `normal_frame/` (clean normal frames, negatives) | **25 / 25 NORMAL** (0 false positives) |

Those are **fit** numbers (the classifier is trained on these same images). The
honest generalisation estimate is its **leave-one-image-out CV**, printed by
`train_camera_clf.py` — on a small training set treat that, not the perfect fit,
as the real-world expectation. Hardest cases: strongly **off-angle** Meta views
and normal **cat-eye** frames whose corner mimics a module. More training images
would tighten this.

---

## Limitations / notes

- **Front-on is the operating condition.** Strongly angled / 3-quarter views may
  read NORMAL.
- **Small training set.** The classifier separates the bundled clean sets
  perfectly but is trained on little data; treat the leave-one-image-out CV as
  the real-world expectation. Add images and re-run `train_camera_clf.py` to
  harden it.
- The locate pipeline (preprocess → segment → locate) is non-ML and
  scale-invariant; it only finds the corner crops. The camera/no-camera call is
  entirely learned, so `models/camera_clf.npz` must be present — `decide.py`
  raises a clear error if it is missing (train it with `train_camera_clf.py`).
- All tunables live in `config.py`; nothing is hard-coded elsewhere.

---

## Layout

The app (runtime):

```
index.html               static page — the whole app, no server-side code
static/opencv.js         vendored OpenCV.js (WASM embedded; self-contained)
static/detector/*.js     JS pipeline: config · hog · camera_clf · preprocess · segment · locate · decide · index
static/camera_clf.json   classifier weights (retrained in the JS-HOG feature space)
static/app.js style.css  UI controller (ES5) + styles
make_and_give.mp4        how-it-works intro clip
```

Python (offline only — train/validate the in-browser model, not needed to run it):

```
detect_meta_glasses.py   CLI entry / reference implementation
config.py                every tunable (single @dataclass)
eval_separation.py       positive/negative separation report
train_camera_clf.py      train the corner camera verifier (LOIO CV + save)
pipeline/                preprocess · segment · locate · features · camera_clf · decide · viz
models/                  camera_clf.npz (cv2-HOG verifier weights)
tools/                   export_clf.py · dump_crops.py · dump_raw.py · train_js_clf.mjs · parity_node.mjs
browser_test.html        real-browser parity check vs Python
ray_ban_frame/           clean Meta frame photos (positives)
normal_frame/            clean normal-frame photos (negatives)
normal_glassess/         worn normal-glasses photos (extra reference)
ray_ban/                 worn Meta photos (extra reference)
debug/                   runtime overlays + JSON (gitignored)
```
