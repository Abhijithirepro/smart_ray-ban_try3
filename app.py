#!/usr/bin/env python3
"""Flask web UI for the Meta-glasses detector.

Upload a clean, roughly front-on photo of a pair of glasses; the server runs the
deterministic CV pipeline and returns the verdict, per-corner scores, the
feature breakdown, and the annotated debug overlay (as a data URI) so the
result can be inspected in the browser.

    python app.py            # then open http://127.0.0.1:5000
"""
from __future__ import annotations

import base64
import io
import os
import tempfile

import cv2
from flask import Flask, jsonify, render_template, request

from config import Config
from detect_meta_glasses import run
from pipeline import viz

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap

CFG = Config()
ALLOWED = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _png_data_uri(bgr_image) -> str:
    """Encode a BGR image to a base64 PNG data URI for inline <img> display."""
    ok, buf = cv2.imencode(".png", bgr_image)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return "data:image/png;base64," + b64


@app.route("/")
def index():
    return render_template("index.html", threshold=CFG.strong_thresh)


@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify({"error": "unsupported file type " + ext}), 400

    # persist to a temp file so the path-based pipeline can read it
    fd, tmp = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(f.read())
        frames, seg, loc, feats, verdict = run(tmp, CFG)
    except Exception as e:  # noqa: BLE001 - surface any decode/pipeline failure
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    overlay = viz.annotate(frames, seg, loc, feats, verdict)
    payload = viz.features_to_dict(feats, loc, seg, verdict)
    payload["overlay"] = _png_data_uri(overlay)
    payload["threshold"] = CFG.strong_thresh
    return jsonify(payload)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
