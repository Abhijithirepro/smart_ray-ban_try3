#!/usr/bin/env python3
"""Meta smart-glasses detector (deterministic classical CV, no ML).

Given a clean, roughly front-on photo of a pair of glasses, decide whether it is
a pair of Ray-Ban Meta / Ray-Ban Stories smart glasses (which carry a small
camera module in a top-outer corner of the frame) versus normal eyeglasses.

    python detect_meta_glasses.py IMAGE [options]

Verdict is the payload; exit code is always 0 unless the image can't be read.
"""
from __future__ import annotations

import argparse
import json
import sys

from config import Config
from pipeline import preprocess, segment, locate, features, decide, viz, facedet


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="path to the glasses photo")
    p.add_argument("--config", help="JSON file of config overrides")
    p.add_argument("--target-width", type=int, help="canonical resize width")
    p.add_argument("--threshold", type=float,
                   help="override cam_clf_thresh (per-corner P(camera) to fire)")
    p.add_argument("--bbox", help="manual frame bbox 'x,y,w,h' (canonical px)")
    p.add_argument("--debug", action="store_true",
                   help="write annotated overlay + features JSON")
    p.add_argument("--debug-dir", default="debug")
    p.add_argument("--json", action="store_true",
                   help="print full feature/score dict instead of one line")
    p.add_argument("--quiet", action="store_true", help="print verdict only")
    return p.parse_args(argv)


def build_config(args) -> Config:
    cfg = Config.from_json(args.config) if args.config else Config()
    if args.target_width:
        cfg.target_width = args.target_width
    if args.threshold is not None:
        cfg.cam_clf_thresh = args.threshold
    return cfg


def run(image_path: str, cfg: Config, manual_bbox=None):
    frames = preprocess.preprocess(image_path, cfg)
    # Worn glasses: anchor on the face (Haar). None -> held-glasses lens path.
    face = None if manual_bbox is not None else facedet.detect_face(frames.gray)
    seg = segment.segment(frames.gray_eq, cfg, manual_bbox=manual_bbox)
    loc = locate.locate(frames.gray_eq, seg, cfg, face=face)
    feats = features.extract(frames, seg, loc, cfg)
    gate_ok, gate_reason = locate.two_lens_gate(loc, cfg)
    verdict = decide.decide(feats, cfg, gate_ok=gate_ok, gate_reason=gate_reason)
    return frames, seg, loc, feats, verdict


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args)

    manual_bbox = None
    if args.bbox:
        manual_bbox = tuple(int(v) for v in args.bbox.split(","))
        if len(manual_bbox) != 4:
            print("error: --bbox needs 'x,y,w,h'", file=sys.stderr)
            return 2

    try:
        frames, seg, loc, feats, verdict = run(args.image, cfg, manual_bbox)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.debug:
        img_path, json_path = viz.save_debug(
            args.image, frames, seg, loc, feats, verdict, args.debug_dir)
        if not args.quiet:
            print(f"debug: {img_path}", file=sys.stderr)
            print(f"debug: {json_path}", file=sys.stderr)

    if args.json:
        print(json.dumps(viz.features_to_dict(feats, loc, seg, verdict), indent=2))
    elif args.quiet:
        print(verdict.verdict)
    else:
        pL = "n/a" if verdict.prob_left is None else f"{verdict.prob_left:.2f}"
        pR = "n/a" if verdict.prob_right is None else f"{verdict.prob_right:.2f}"
        print(f"{verdict.verdict}  Pcam L={pL} R={pR}  ({args.image})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
