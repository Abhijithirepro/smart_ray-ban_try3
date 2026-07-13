#!/usr/bin/env python3
"""Minimal local box-review server for data/boxes/boxes.json.

The bootstrapper (tools/bootstrap_boxes.py) auto-places one whole-glasses box per
positive using the Haar-anchored localizer. It is good on frontal "worn" shots but
loose on studio "held" product photos and on the "center" fallback (no face found).
This tiny tool lets you eyeball every positive and drag/resize/delete/redraw its box,
saving corrections straight back into boxes.json.

    python tools/label_review/serve.py           # then open http://localhost:8765
    python tools/label_review/serve.py --port 9000 --all   # also review negatives

Pure stdlib (http.server) — no Flask, no deps. Serves images from data/boxes/ and
exposes two JSON endpoints:
    GET  /api/manifest          -> boxes.json (+ index metadata)
    POST /api/save              -> {image, box:[x,y,w,h] | null}  (rewrites boxes.json)
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
BOXES_DIR = os.path.join(_REPO, "data", "boxes")
MANIFEST = os.path.join(BOXES_DIR, "boxes.json")
INDEX_HTML = os.path.join(_HERE, "index.html")

_LOCK = threading.Lock()
SHOW_ALL = False

_CTYPE = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
          ".html": "text/html; charset=utf-8", ".js": "application/javascript"}


def _load():
    with open(MANIFEST) as fh:
        return json.load(fh)


def _save(manifest):
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=1)
    os.replace(tmp, MANIFEST)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            with open(INDEX_HTML, "rb") as fh:
                return self._send(200, fh.read(), _CTYPE[".html"])
        if path == "/api/manifest":
            with _LOCK:
                m = _load()
            # attach a stable review order (positives first unless --all)
            imgs = [r for r in m["images"] if SHOW_ALL or r["label"] == 1]
            return self._send(200, {"target_width": m.get("target_width"),
                                    "show_all": SHOW_ALL, "images": imgs})
        # static image files under data/boxes/
        rel = path.lstrip("/")
        fpath = os.path.normpath(os.path.join(BOXES_DIR, rel))
        if not fpath.startswith(BOXES_DIR) or not os.path.isfile(fpath):
            return self._send(404, {"error": "not found"})
        ext = os.path.splitext(fpath)[1].lower()
        with open(fpath, "rb") as fh:
            return self._send(200, fh.read(), _CTYPE.get(ext, "application/octet-stream"))

    def do_POST(self):
        if urlparse(self.path).path != "/api/save":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        image = payload.get("image")
        box = payload.get("box")  # [x,y,w,h] ints or null
        if box is not None:
            box = [int(round(v)) for v in box]
        with _LOCK:
            m = _load()
            hit = None
            for r in m["images"]:
                if r["image"] == image:
                    r["box"] = box
                    r["reviewed"] = True
                    hit = r
                    break
            if hit is None:
                return self._send(404, {"error": f"no such image: {image}"})
            _save(m)
        return self._send(200, {"ok": True, "image": image, "box": box})


def main(argv=None):
    global SHOW_ALL
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--all", action="store_true",
                    help="also list negatives (to catch mislabeled backgrounds)")
    args = ap.parse_args(argv)
    SHOW_ALL = args.all

    if not os.path.isfile(MANIFEST):
        raise SystemExit(f"manifest not found: {MANIFEST}\n"
                         "run tools/bootstrap_boxes.py first")

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"box review UI  ->  http://localhost:{args.port}")
    print("keys: →/Enter save+next, ← prev, d delete box, n new box, r reset. Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
