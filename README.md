# MODULE SCAN — Meta smart-glasses camera detector

A small, transparent, **deterministic classical-CV** tool that decides whether a
clean, roughly front-on photo of a pair of glasses is a pair of **Ray-Ban Meta /
Ray-Ban Stories** smart glasses (which carry a small camera module in a
top-outer corner of the frame) versus **normal eyeglasses**.

**No machine learning of any kind** — no neural nets, no trained classifiers, no
Haar cascades. Only thresholding, contours, Hough circles, distance transforms
and hand-written geometric heuristics.

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
```

| module | does |
|--------|------|
| `pipeline/preprocess.py` | load (OpenCV→Pillow fallback, flattens transparent PNGs onto white), canonical resize, grayscale + CLAHE + blur variants |
| `pipeline/segment.py` | frame bbox + filled mask via inverse-Otsu → morphology → largest contour, with a Canny fallback and a sanity gate; flags whether an **isolated** frame was found |
| `pipeline/locate.py` | derive a **generous top-outer search region** per side, relative to the frame bbox (scales with the frame, so near/far is handled) |
| `pipeline/features.py` | per corner: gather circle / dark-blob / glint candidates, refine each to its local bezel, and pick the most **camera-like** one (bezel + central glint + dark core); emit the features |
| `pipeline/decide.py` | per-corner weighted score → fire META if **either** corner clears the threshold; **domain gate**: abstain (NORMAL) if no isolated glasses frame was found |
| `pipeline/viz.py` | annotated overlay (bbox, ROIs, candidate/chosen circles, glint pixels, scores) + features JSON |

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
| `ray_ban_frame/` (clean Meta frames, positives) | **6 / 9 META** |
| `normal_glassess/` (worn normal glasses, negatives) | **19 / 19 NORMAL** (0 false positives) |

- **Front-on Meta frames: 6 / 6 caught.**
- The 3 misses are **off-angle views** (`al3`, `qt`, `mew`) where the camera is
  foreshortened into an ellipse and the circular signature weakens (scores
  0.55–0.59, just under the 0.60 threshold). These are outside the
  "roughly front-on" operating condition and are a known, improvable limitation.

---

## Limitations / notes

- **Front-on is the operating condition.** Strongly angled / 3-quarter views may
  read NORMAL.
- **Negatives are worn-on-face photos**, a different domain from the clean
  positive frames. They're rejected mostly by the domain gate. A set of **clean
  normal-frame product photos** (plain background, just the glasses) would let
  the decision margin be calibrated properly against the real operating
  condition — currently that margin is thin (front positives ≥ 0.64; the few
  cleanly-segmented normals sit ≤ 0.55).
- Heuristic and non-ML by design — inherently more brittle than a trained
  detector. That trade-off is intentional.
- All thresholds live in `config.py`; nothing is hard-coded elsewhere.

---

## Layout

```
detect_meta_glasses.py   CLI entry
app.py                   Flask web app
config.py                every tunable (single @dataclass)
eval_separation.py       positive/negative separation report
pipeline/                preprocess · segment · locate · features · decide · viz
templates/ static/       web UI (Jinja template, ES5 JS, CSS)
ray_ban_frame/           clean Meta frame photos (positives)
normal_glassess/         worn normal-glasses photos (negatives)
ray_ban/                 worn Meta photos (extra reference)
debug/                   runtime overlays + JSON (gitignored)
```
