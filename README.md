# MODULE SCAN — Meta smart-glasses camera detector

A small, transparent tool that decides whether a clean, roughly front-on photo
of a pair of glasses is a pair of **Ray-Ban Meta / Ray-Ban Stories** smart
glasses (which carry a small camera module in a top-outer corner of the frame)
versus **normal eyeglasses**.

The geometry is classical CV — thresholding, contours, Hough circles, distance
transforms, segmentation. But telling a *camera lens* apart from a *rounded
frame corner* is a structural/semantic distinction that classical circle and
darkness cues provably cannot make (both are "a dark circle in the corner"), so
a **tiny, self-contained classifier** does that last step: a HOG descriptor over
the corner crop feeds an L2-regularised logistic regression (pure numpy + the
OpenCV HOG, no external ML dependency, weights in `models/camera_clf.npz`). It
verifies each corner; the rest of the pipeline is unchanged.

---

## The signal it looks for

On a Ray-Ban Meta frame the camera appears as a **small, dark, glassy,
near-circular module** set in a slightly raised metallic **bezel**, usually with
a tiny bright **specular glint** at its centre, in the **top-outer corner / end
piece** where the top rim meets the temple hinge.

The latest Ray-Ban Meta (Gen 2) has **one** camera in one top-outer corner; the
older Ray-Ban Stories has one per corner. So the detector scans **both**
top-outer corners and **fires if either** looks like a camera — robust to
photo flip/orientation and to one- vs two-camera models. The privacy LED is
**never** used as a signal.

A corner reads as a camera when it shows the **full coincident signature**:
a bezel **circle** + a **central glint** + a **dark glassy core**, in a **chunky**
corner. Distractors (a lens edge, a screw, a reflection, an eye) tend to have
only one or two of these, not all at the right scale and place.

---

## Install

```bash
cd try3-non-ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt        # numpy, Pillow, opencv-python-headless
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
META  overall=0.75  L=0.75 R=0.50  (ray_ban_frame/0rw4013__6702ce__p21__shad__fr.png)
```

Options:

| flag | meaning |
|------|---------|
| `--debug` | write an annotated overlay + features JSON to `--debug-dir` (default `debug/`) |
| `--debug-dir DIR` | where debug artefacts go |
| `--json` | print the full feature/score dict instead of the one-liner |
| `--config FILE` | JSON file of config overrides (see `config.py`) |
| `--target-width N` | canonical resize width (default 1000) |
| `--threshold T` | override the META fire threshold |
| `--bbox x,y,w,h` | manual frame bounding box (skip segmentation) |
| `--require-both` | require a camera in **both** corners (Stories-style) |
| `--quiet` | print just the verdict word |

The verdict is the payload; exit code is `0` unless the image can't be read.

## Use it — web UI

```bash
python app.py          # then open http://127.0.0.1:5000
```

Drag a photo (or click) onto the scanner. It shows the **verdict**, a
**confidence meter** with the decision threshold marked, the **per-corner
scores**, the lit **signature chips** (bezel / glint / dark core / chunky), and
the **annotated overlay** so you can see exactly what fired and where.

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
| `pipeline/features.py` | per corner: gather circle / dark-blob / glint candidates, refine each to its local bezel, pick the most **camera-like** one, emit the features, and attach the learned **`cam_prob`** for the corner crop |
| `pipeline/camera_clf.py` | the learned verifier: corner crop → 32×32 (R mirrored to canonical) → illumination-normalise → HOG → standardise → logistic regression → `P(camera)` |
| `pipeline/decide.py` | fire META iff **both** corners are cameras (`cam_prob ≥ cam_clf_thresh`; falls back to Hough-circle presence if no model file). **Domain gate**: abstain (NORMAL) if no isolated glasses frame was found. The old weighted score is kept for debug only |
| `pipeline/viz.py` | annotated overlay (bbox, ROIs, candidate/chosen circles, glint pixels, per-corner `Pcam`) + features JSON |

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
| `ray_ban_frame/` (clean Meta frames, positives) | **9 / 9 META** |
| `normal_frame/` (clean normal frames, negatives) | **10 / 10 NORMAL** (0 false positives) |

Those are **fit** numbers (the verifier is trained on these same images). The
honest generalisation estimate is the verifier's **leave-one-image-out CV: ~84 %**
(printed by `train_camera_clf.py`) — with only 19 training images, treat that,
not the 19/19, as the real-world expectation. Hardest cases: strongly
**off-angle** Meta views and normal **cat-eye** frames whose corner mimics a
module. More training images would tighten this.

---

## Limitations / notes

- **Front-on is the operating condition.** Strongly angled / 3-quarter views may
  read NORMAL.
- **Tiny training set (19 images).** The verifier separates the bundled clean
  sets perfectly but is trained on very little data; expect the ~84 % LOIO-CV to
  be closer to real-world performance. Add images and re-run
  `train_camera_clf.py` to harden it.
- The geometry pipeline is non-ML and scale-invariant; only the final
  per-corner camera/no-camera call is learned. If `models/camera_clf.npz` is
  absent, `decide.py` falls back to raw Hough-circle presence (which **does**
  false-positive on rounded normal frames — the reason the verifier exists).
- All thresholds live in `config.py`; nothing is hard-coded elsewhere.

---

## Layout

```
detect_meta_glasses.py   CLI entry
app.py                   Flask web app
config.py                every tunable (single @dataclass)
eval_separation.py       positive/negative separation report
train_camera_clf.py      train the corner camera verifier (LOIO CV + save)
pipeline/                preprocess · segment · locate · features · camera_clf · decide · viz
models/                  camera_clf.npz (trained verifier weights)
templates/ static/       web UI (Jinja template, ES5 JS, CSS)
ray_ban_frame/           clean Meta frame photos (positives)
normal_frame/            clean normal-frame photos (negatives)
normal_glassess/         worn normal-glasses photos (extra reference)
ray_ban/                 worn Meta photos (extra reference)
debug/                   runtime overlays + JSON (gitignored)
```
