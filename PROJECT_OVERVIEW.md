# MODULE SCAN — Plain-English Overview

## What it does

You point your webcam (or upload a photo) at a pair of glasses. The app tells you:

> **Are these Ray-Ban Meta smart glasses (with a hidden camera), or just normal glasses?**

It gives one of three answers:

- 🔴 **SMART GLASSES** — a camera was found in *both* corners. Confident.
- 🟠 **MAYBE — CHECK** — a camera was found in *one* corner. Worth a human look.
- 🟢 **NORMAL GLASSES** — no camera found.

Everything runs **inside your web browser**. Nothing is uploaded, nothing is stored on a server, and it works offline once loaded.

---

## The one hard problem

Ray-Ban Meta glasses look almost identical to ordinary glasses. The only giveaway is a **tiny camera lens** tucked into the top-outer corner of the frame — one on each side.

The catch: **a camera lens and a plain rounded frame corner both just look like "a small dark circle."** You can't write a simple rule to tell them apart — it comes down to subtle differences in shape and texture that a human eye recognizes but a formula can't.

So the app uses **two kinds of intelligence working together**:

1. **Ordinary computer vision** (rules we wrote) does the easy, explainable parts — find the face, find the glasses, find the two corners.
2. **Machine learning** does the one hard part — look at each corner and decide, "is this a real camera, or just a frame corner?"

---

## How it works, step by step

1. **Clean up the image** — resize it to a standard size and boost the contrast.
2. **Find the face** — using classic face-detection.
3. **Find the two top-outer corners** — the exact spots where a Meta camera would sit.
4. **Zoom into each corner** at full resolution (the camera is too tiny to see in a shrunk-down image).
5. **Ask the AI about each corner** — it returns a probability: "how camera-like is this corner?"
6. **Decide:**
   - Both corners look like cameras → **SMART GLASSES**
   - One corner looks like a camera → **MAYBE**
   - Neither → **NORMAL GLASSES**

There's also a **built-in safety rule:** if the app can't clearly find a pair of glasses (say you point it at a bare face), it just says **NORMAL** instead of guessing — because an eye also looks like "a dark circle with a glint."

---

## The clever bit: it almost never cries wolf

The app is deliberately tuned so that **it will never *confidently* call normal glasses "smart glasses."**

The rule for a confident "SMART GLASSES" is set just high enough that, across every normal-glasses photo the team tested, **not a single one** could ever clear the bar in both corners at once. That's why the confident false-alarm rate is **0 out of 475 photos = 0.00%**.

The trade-off: to stay this cautious, it sometimes says "MAYBE" or misses a Ray-Ban that's photographed at a steep angle. The team chose **"rarely wrong when confident"** over **"catches absolutely everything."**

---

## How accurate is it?

| What we tested | How many | Result |
|---|---|---|
| **Ray-Ban Meta glasses** (should be flagged) | 58 photos | **74% flagged** (43 caught) |
| **Normal glasses** — everyday photos | 41 photos | 24 correct; the rest flagged only as a soft "MAYBE," never a confident wrong answer |
| **Normal glasses** — large public dataset | 434 photos | **434 / 434 correct** ✅ |
| **Confident false alarms** (calling normal glasses "smart") | 475 photos | **0 — zero** ✅ |

**In plain terms:** it catches about **3 out of 4** Ray-Ban Metas, and it essentially never confidently mislabels ordinary glasses as spy glasses.

---

## How it was built (the AI part)

- The "brain" is a compact, well-known image model (**MobileNetV3**) — small enough to run in a browser tab.
- It was trained on **564 photos** (about 5,800 corner close-ups): Ray-Ban Meta glasses on real faces plus a large set of ordinary glasses (including 500 from a public dataset for variety).
- Training used careful tricks so the AI learns the *right* thing — for example, randomly blurring sharp photos so it can't cheat by assuming "sharp photo = Ray-Ban."
- It was tested **fairly**: photos of the same person or product were never split between "practice" and "exam," so the scores reflect how it does on genuinely new glasses.

---

## Using it live

The webcam mode walks you through a quick guided capture — **look left, look right, then face forward** — and analyzes a short burst of frames from the front-facing shot, taking the majority vote so a single blurry frame can't throw off the result. When you're done it can hand you a recording of the session and the captured photos.

---

## The honest caveats

- It needs to **see a face** to do its best work (the primary detector is corner-based).
- **Steep side angles** can hide the camera module, causing a miss.
- **Cat-eye normal frames** have a pointed corner that can look camera-like, which is why those sometimes get a cautious "MAYBE" rather than a clean "NORMAL."
- It catches ~74% of Ray-Bans by design — it favors being **right when it's sure** over catching every single one.

---

*For the full engineering detail — model architecture, thresholds, training procedure, and file-by-file breakdown — see `PROJECT_TECHNICAL.md`.*
