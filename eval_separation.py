#!/usr/bin/env python3
"""Evaluate separation between the positive (Meta frame) and negative sets.

Prints the per-image max-corner score for each set and the threshold that best
separates them. This is the tuning instrument: optimise for SEPARATION, not for
hitting any particular number.

    python eval_separation.py [--debug]
"""
import glob
import sys

from config import Config
from pipeline import preprocess, segment, locate, features, decide

POS = ("ray_ban_frame", "META")       # clean Meta frame shots  -> want META
NEG = ("normal_frame", "NORMAL")      # clean normal frame shots -> want NORMAL


def score(path, cfg):
    fr = preprocess.preprocess(path, cfg)
    seg = segment.segment(fr.gray_eq, cfg)
    loc = locate.locate(fr.gray_eq, seg, cfg)
    feats = features.extract(fr, seg, loc, cfg)
    v = decide.decide(feats, cfg, seg=seg, shape=fr.gray.shape)
    return v, feats


def _pcam(feats):
    pL, pR = feats["L"].cam_prob, feats["R"].cam_prob
    if pL is None or pR is None:
        return "Pcam=n/a    "
    return f"Pcam L={pL:.2f} R={pR:.2f}"


def files_in(folder):
    fs = sorted(glob.glob(f"{folder}/*"))
    return [f for f in fs if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]


def main():
    cfg = Config()
    pos_meta = neg_meta = 0
    pos_n = neg_n = 0
    print(f"\n=== POSITIVES {POS[0]} (want META) ===")
    for f in files_in(POS[0]):
        v, feats = score(f, cfg); pos_n += 1
        meta = v.verdict == "META"; pos_meta += meta
        print(f"  {v.verdict:7s} {_pcam(feats)}  "
              f"{f.split('/')[-1]}  {'' if meta else '<-- MISS'}")
    print(f"\n=== NEGATIVES {NEG[0]} (want NORMAL) ===")
    for f in files_in(NEG[0]):
        v, feats = score(f, cfg); neg_n += 1
        meta = v.verdict == "META"; neg_meta += meta
        print(f"  {v.verdict:7s} {_pcam(feats)}  "
              f"{f.split('/')[-1]}  {'<-- FALSE POS' if meta else ''}")

    print("\n--- summary ---")
    print(f"positives caught : {pos_meta}/{pos_n} META")
    print(f"negatives rejected: {neg_n - neg_meta}/{neg_n} NORMAL "
          f"({neg_meta} false positives)")
    print(f"overall accuracy : {pos_meta + (neg_n - neg_meta)}/{pos_n + neg_n}")


if __name__ == "__main__":
    main()
