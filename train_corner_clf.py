"""Train the top-outer camera-MODULE CNN (Ray-Ban vs normal), per corner.

Where train_region_clf classified a whole-face crop (and could not see the tiny
camera module, so it guessed from pose), this classifies a full-resolution
TOP-OUTER CORNER crop, where the module is actually resolvable. A Ray-Ban carries
a module in BOTH corners, so the image verdict is the MIN of its two corner
probabilities — both end-pieces must look like a module to fire META. That both
rejects normal glasses (bare rim on both sides) and defeats the pose shortcut (a
hand near one side cannot fake a module on both specific corners).

MobileNetV3-small (ImageNet) fine-tuned on corner crops; grouped K-fold CV and
FP-first thresholding operate at IMAGE level on the aggregated probability.

    python train_corner_clf.py
Env knobs: EPOCHS_HEAD, EPOCHS_FT, KFOLDS, BATCH, INPUT (default 112),
           FP_BUDGET_BULK, TARGET_RECALL, DUMP_ERRORS.
"""
from __future__ import annotations

import json
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision as tv
from torchvision.transforms import v2 as T

from config import Config
from datasets_corners import load_corner_dataset

SEED = 0
INPUT = int(os.environ.get("INPUT", 112))
BATCH = int(os.environ.get("BATCH", 32))
EPOCHS_HEAD = int(os.environ.get("EPOCHS_HEAD", 12))
EPOCHS_FT = int(os.environ.get("EPOCHS_FT", 8))
KFOLDS = int(os.environ.get("KFOLDS", 4))
MODEL_OUT = "models/corner_clf.pt"
OOF_REPORT = "models/corner_oof_report.json"

FP_BUDGET_BULK = float(os.environ.get("FP_BUDGET_BULK", 0.01))
TARGET_RECALL = float(os.environ.get("TARGET_RECALL", 0.90))
DUMP_ERRORS = os.environ.get("DUMP_ERRORS") == "1"
MEGLASS_SOURCE = "data/normal/meglass"

# Keep the machine busy during the training epochs: the per-sample augmentation
# (RandomResizedCrop/DownscaleJitter/ColorJitter/rotate) is CPU work that otherwise
# serialises with the MPS forward/backward. Worker processes overlap it with GPU
# compute. persistent_workers amortises the spawn/import cost across the 20 epochs
# of each fold. Set NUM_WORKERS=0 to fall back to the exact single-process recipe.
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", min(4, max(0, (os.cpu_count() or 2) - 2))))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

class DownscaleJitter(torch.nn.Module):
    """Randomly downscale to 48..INPUT px and back (p=0.3). The MeGlass bulk
    negatives are 120px-source (blurry) while positives are sharp full-res crops —
    without this, 'sharp = module' is a free shortcut. Blur-jittering all training
    crops makes sharpness class-uninformative and also helps low-res Ray-Bans."""
    def __init__(self, lo=48, hi=INPUT, p=0.3):
        super().__init__()
        self.lo, self.hi, self.p = lo, hi, p

    def forward(self, img):
        if torch.rand(1).item() >= self.p:
            return img
        side = int(torch.randint(self.lo, self.hi + 1, (1,)).item())
        small = T.functional.resize(img, [side, side],
                                    interpolation=T.InterpolationMode.BILINEAR,
                                    antialias=True)
        return T.functional.resize(small, [self.hi, self.hi],
                                   interpolation=T.InterpolationMode.BICUBIC,
                                   antialias=True)


# No horizontal flip: corners are already mirrored to one canonical orientation,
# so flipping would fight that convention. Module can sit slightly in/out of frame,
# hence the mild scale/translation jitter via RandomResizedCrop.
_train_tf = T.Compose([
    T.ToImage(),
    T.Resize((int(INPUT * 1.15), int(INPUT * 1.15))),
    T.RandomResizedCrop(INPUT, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
    DownscaleJitter(),
    T.ColorJitter(0.3, 0.3, 0.3, 0.05),
    T.RandomRotation(8),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
_eval_tf = T.Compose([
    T.ToImage(),
    T.Resize((INPUT, INPUT)),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class CornerDS(torch.utils.data.Dataset):
    def __init__(self, items, train):
        self.items = items
        self.tf = _train_tf if train else _eval_tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        x = self.tf(Image.fromarray(it.rgb))
        return x, torch.tensor([float(it.label)])


def build_model():
    m = tv.models.mobilenet_v3_small(
        weights=tv.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_f = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_f, 1)
    return m.to(DEVICE)


def _set_backbone_grad(m, on):
    for p in m.features.parameters():
        p.requires_grad = on


def _tier(it):
    if it.label == 1:
        return "pos"
    return "bulk" if it.source == MEGLASS_SOURCE else "indomain"


# Sampler mass per tier. In-domain worn normals dominate the negative mass; the
# 120px MeGlass bulk keeps only a small share (its blur made it a shortcut, see
# DownscaleJitter) — do NOT raise it back without full-res source images.
SAMPLER_MASS = {"pos": 0.5, "indomain": 0.4, "bulk": 0.1}


def _make_sampler(items, gen=None):
    """WeightedRandomSampler with SAMPLER_MASS per tier (renormalised over the
    tiers actually present). Keeps the worn normals in most negative batches so
    the bulk set can't create a domain shortcut."""
    tiers = [_tier(it) for it in items]
    counts = {t: tiers.count(t) for t in set(tiers)}
    target = {t: m for t, m in SAMPLER_MASS.items() if counts.get(t)}
    tot = sum(target.values()) or 1.0
    weights = [target[t] / tot / counts[t] for t in tiers]
    return torch.utils.data.WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double), num_samples=len(items),
        replacement=True, generator=gen)


def train_one(train_items, seed=SEED):
    """Train one model. Only crops flagged `train_use` are trained on: positive
    CENTER boxes (minus excluded sides) + all NEGATIVE grid crops (hard negatives).
    Positive grid crops and excluded sides are eval-only. Seeded so each fold's RNG
    is independent of how much randomness earlier folds consumed."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    train_items = [it for it in train_items if getattr(it, "train_use", True)]
    gen = torch.Generator().manual_seed(seed)
    m = build_model()
    dl = torch.utils.data.DataLoader(CornerDS(train_items, True), batch_size=BATCH,
                                     sampler=_make_sampler(train_items, gen),
                                     num_workers=NUM_WORKERS,
                                     persistent_workers=(NUM_WORKERS > 0),
                                     prefetch_factor=(2 if NUM_WORKERS > 0 else None))
    crit = nn.BCEWithLogitsLoss()
    _set_backbone_grad(m, False)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(EPOCHS_HEAD):
        _epoch(m, dl, crit, opt)
    _set_backbone_grad(m, True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-4)
    for _ in range(EPOCHS_FT):
        _epoch(m, dl, crit, opt)
    return m


def _epoch(m, dl, crit, opt):
    m.train()
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = crit(m(x), y)
        loss.backward()
        opt.step()


@torch.no_grad()
def predict(m, items):
    m.eval()
    dl = torch.utils.data.DataLoader(CornerDS(items, False), batch_size=BATCH)
    out = []
    for x, _ in dl:
        p = torch.sigmoid(m(x.to(DEVICE))).cpu().numpy().ravel()
        out.extend(p.tolist())
    return np.array(out)


# Per-side aggregation over the candidate grid. "gridmax" = best box (recall-first,
# but a lone spurious box inflates normals). "second_high" = 2nd-highest GRID box
# (a real module fires >=2 overlapping boxes; a lone spurious normal patch fires 1)
# — parameter-free FP suppressor. Applied over grid boxes only (box_idx>=0); the
# center box (box_idx=-1) duplicates a grid combo and is excluded from the stat.
AGGREGATE = os.environ.get("AGGREGATE", "gridmax")   # "gridmax" | "second_high"


def _side_score(box_probs):
    """box_probs: list of (box_idx, prob). Reduce to one per-side score using the
    grid boxes (box_idx>=0). gridmax=max; second_high=2nd largest (>=2 boxes must
    agree)."""
    grid = sorted((p for bi, p in box_probs if bi >= 0), reverse=True)
    if not grid:
        allp = [p for _, p in box_probs]
        return max(allp) if allp else 0.0
    if AGGREGATE == "second_high":
        return grid[1] if len(grid) >= 2 else grid[0]
    return grid[0]


def aggregate_images(items, corner_probs):
    """Reduce per-CROP probs to one [L, R] pair per image via the per-side grid
    statistic (_side_score). The three-tier verdict reads the two per-side scores —
    YES = both sides >= tc_hi, MAYBE = one >= tc_maybe. Raw per-box vectors are kept
    (boxes_L/boxes_R) so aggregation choices can be re-evaluated offline from one
    CV run. Returns dict image_id -> {probs:[L,R], prob, boxes_L, boxes_R, label,
    dist, source, group, path}."""
    by_img = {}
    for it, p in zip(items, corner_probs):
        d = by_img.setdefault(it.image_id, {"L": [], "R": [], "label": it.label,
                                            "dist": it.dist, "source": it.source,
                                            "group": it.group, "path": it.path})
        d[it.side].append((it.box_idx, float(p)))
    for d in by_img.values():
        sL = _side_score(d["L"]); sR = _side_score(d["R"])
        # store raw grid-box vectors (box_idx>=0), sorted desc, for offline sweeps
        d["boxes_L"] = sorted((p for bi, p in d["L"] if bi >= 0), reverse=True)
        d["boxes_R"] = sorted((p for bi, p in d["R"] if bi >= 0), reverse=True)
        d["probs"] = [sL, sR]
        d["prob"] = max(sL, sR)
    return by_img


def grouped_folds(items, k):
    groups = {}
    for it in items:
        groups.setdefault(it.group, it.label)
    gl = list(groups.items())
    rng = np.random.RandomState(SEED)
    rng.shuffle(gl)
    fold_of, counters = {}, {0: 0, 1: 0}
    for g, lab in gl:
        fold_of[g] = counters[lab] % k
        counters[lab] += 1
    return fold_of


def metrics(probs, labels, thr):
    probs, labels = np.asarray(probs), np.asarray(labels)
    pred = (probs >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    npos = max(1, (labels == 1).sum()); nneg = max(1, (labels == 0).sum())
    return {"acc": (tp + tn) / max(1, len(labels)), "recall": tp / npos,
            "fp_rate": fp / nneg, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _recall_first_threshold(probs, labels, target_recall=0.90):
    ts = np.linspace(0.05, 0.95, 37)
    best = None
    for t in ts:
        if metrics(probs, labels, t)["recall"] >= target_recall:
            best = t
    return float(best) if best is not None else float(ts[0])


def pick_threshold(probs, labels, dists, sources,
                   fp_budget_bulk=FP_BUDGET_BULK, target_recall=TARGET_RECALL):
    """FP-first image-level threshold on the REAL (worn) subset. Clears every
    in-domain negative (Floor A -> zero worn FP) and fp_budget_bulk of MeGlass
    (Floor B); sits at the class-gap midpoint when one exists, else on the floor."""
    probs = np.asarray(probs, float); labels = np.asarray(labels, int)
    dists = np.asarray(dists); sources = np.asarray(sources)
    real = dists == "real"
    p, l, s = probs[real], labels[real], sources[real]
    neg_in = p[(l == 0) & (s != MEGLASS_SOURCE)]
    neg_bulk = p[(l == 0) & (s == MEGLASS_SOURCE)]
    pos = p[l == 1]
    floor_a = float(np.nextafter(neg_in.max(), 1.0)) if neg_in.size else 0.0
    floor_b = float(np.quantile(neg_bulk, 1.0 - fp_budget_bulk)) if neg_bulk.size else 0.0
    q_neg = max(floor_a, floor_b)
    q_pos = float(np.quantile(pos, 1.0 - target_recall)) if pos.size else q_neg
    thr = 0.5 * (q_neg + q_pos) if q_pos > q_neg else q_neg
    # ceiling 0.999 (not 0.99): with the strict FP-first floor a hard negative can
    # legitimately sit at 0.995, and clamping below it would re-admit that FP.
    return float(min(0.999, max(0.05, thr)))


def pick_tier_thresholds(imgs, review_budget=0.10):
    """Structural three-tier thresholds from the worn in-domain OOF distributions.

    YES  = both corners >= tc_hi   (confident: straight-on, both modules seen)
    MAYBE= one corner  >= tc_maybe (profile / one side occluded -> flag for check)
    NO   = otherwise

    tc_hi clears every in-domain normal's SECOND-highest corner, so no normal can
    ever collect two hits -> zero confident FPs BY CONSTRUCTION (a single weird
    corner on a normal can only ever reach MAYBE). tc_maybe is the smallest t
    whose MAYBE review load on normals stays within `review_budget` (default 10%,
    i.e. <=ceil(0.10*n) normals flagged), clamped to [0.5, tc_hi]. MeGlass is
    excluded from derivation (low-res bulk) but reported as sanity elsewhere."""
    real_in = [d for d in imgs.values() if d["dist"] == "real"
               and d["source"] != MEGLASS_SOURCE]
    negs = [d for d in real_in if d["label"] == 0]
    if not negs:
        return 0.9, 0.5
    second = [sorted(d["probs"])[-2] if len(d["probs"]) >= 2 else min(d["probs"])
              for d in negs]
    tc_hi = float(np.nextafter(max(second), 1.0))
    tc_hi = min(tc_hi, 0.999)
    budget = int(np.ceil(review_budget * len(negs)))
    neg_max = np.array([max(d["probs"]) for d in negs])
    tc_maybe = tc_hi
    for t in np.linspace(0.5, tc_hi, 200):
        if int((neg_max >= t).sum()) <= budget:
            tc_maybe = float(t)
            break
    return tc_hi, tc_maybe


def tier_verdict(probs, tc_hi, tc_maybe):
    """'YES' | 'MAYBE' | 'NO' for one image's corner probs."""
    if sum(1 for p in probs if p >= tc_hi) >= 2:
        return "YES"
    if any(p >= tc_maybe for p in probs):
        return "MAYBE"
    return "NO"


def _tier_table(imgs, tc_hi, tc_maybe):
    """Print the YES/MAYBE/NO breakdown per group at the chosen tier thresholds."""
    print(f"  three-tier: tc_hi={tc_hi:.3f} (YES=both corners)  "
          f"tc_maybe={tc_maybe:.3f} (MAYBE=one corner)")
    rows = (("RAYBAN worn", lambda d: d["label"] == 1 and d["dist"] == "real"),
            ("NORMAL in-domain", lambda d: d["label"] == 0 and d["dist"] == "real"
             and d["source"] != MEGLASS_SOURCE),
            ("NORMAL meglass", lambda d: d["source"] == MEGLASS_SOURCE))
    out = {}
    for name, pred in rows:
        g = [d for d in imgs.values() if pred(d)]
        y = sum(1 for d in g if tier_verdict(d["probs"], tc_hi, tc_maybe) == "YES")
        mb = sum(1 for d in g if tier_verdict(d["probs"], tc_hi, tc_maybe) == "MAYBE")
        n = len(g) - y - mb
        out[name] = {"YES": y, "MAYBE": mb, "NO": n, "total": len(g)}
        extra = f"  flagged={100*(y+mb)/max(1,len(g)):.0f}%" if "RAYBAN" in name else ""
        print(f"    {name:18} YES={y:<3} MAYBE={mb:<3} NO={n:<4} of {len(g)}{extra}")
    return out


def _tradeoff_table(imgs):
    real = [d for d in imgs.values() if d["dist"] == "real"]
    P = np.array([d["prob"] for d in real if d["label"] == 1])
    NI = np.array([d["prob"] for d in real if d["label"] == 0 and d["source"] != MEGLASS_SOURCE])
    print(f"  worn images: {len(P)} rayban, {len(NI)} in-domain normal")
    print("  thr   recall(rayban)   FP(in-domain normal)")
    for t in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        rec = (P >= t).mean() * 100 if len(P) else 0
        fp = int((NI >= t).sum())
        print(f"    {t:.2f}    {rec:5.1f}%          {fp}/{len(NI)}")


def main():
    cfg = Config()
    print(f"device={DEVICE} input={INPUT} loading corner dataset ...")
    items = load_corner_dataset(cfg)
    n_img = len({it.image_id for it in items})
    labels_c = np.array([it.label for it in items])
    print(f"{len(items)} corner crops from {n_img} images | "
          f"rayban_corners={(labels_c==1).sum()} normal_corners={(labels_c==0).sum()} | "
          f"groups={len({it.group for it in items})}")

    fold_of = grouped_folds(items, KFOLDS)
    oof = np.full(len(items), np.nan)
    for k in range(KFOLDS):
        test_idx = [i for i, it in enumerate(items) if fold_of[it.group] == k]
        train_idx = [i for i in range(len(items)) if i not in set(test_idx)]
        if not test_idx:
            continue
        m = train_one([items[i] for i in train_idx], seed=SEED * 1000 + k)
        oof[test_idx] = predict(m, [items[i] for i in test_idx])
        print(f"  fold {k+1}/{KFOLDS}: trained {len(train_idx)} corners, tested {len(test_idx)}")

    valid = ~np.isnan(oof)
    imgs = aggregate_images([items[i] for i in range(len(items)) if valid[i]],
                            oof[valid])
    ip = np.array([d["prob"] for d in imgs.values()])
    il = np.array([d["label"] for d in imgs.values()])
    idist = np.array([d["dist"] for d in imgs.values()])
    isrc = np.array([d["source"] for d in imgs.values()])

    thr = pick_threshold(ip, il, idist, isrc)
    real = idist == "real"
    old_thr = _recall_first_threshold(ip[real], il[real])

    print(f"\n=== GROUPED {KFOLDS}-FOLD CV (image-level, aggregate={AGGREGATE.upper()}) | threshold={thr:.3f} ===")
    print(f"  policy: FP-first (fp_budget_bulk={FP_BUDGET_BULK}, target_recall={TARGET_RECALL})")
    overall = metrics(ip, il, thr)
    print(f"  overall: acc={overall['acc']*100:.1f}%  recall(rayban)={overall['recall']*100:.1f}%  "
          f"FP-rate={overall['fp_rate']*100:.1f}%  [tp={overall['tp']} fp={overall['fp']} tn={overall['tn']} fn={overall['fn']}]")
    report_metrics = {"overall": overall}
    for d in ("studio", "real"):
        idx = idist == d
        if idx.any():
            mm = metrics(ip[idx], il[idx], thr)
            report_metrics[d] = mm
            print(f"  {d:6}: acc={mm['acc']*100:.1f}%  recall={mm['recall']*100:.1f}%  "
                  f"FP-rate={mm['fp_rate']*100:.1f}%  [tp={mm['tp']} fp={mm['fp']} tn={mm['tn']} fn={mm['fn']}]")
    r_new = metrics(ip[real], il[real], thr)["recall"]
    r_old = metrics(ip[real], il[real], old_thr)["recall"]
    print(f"  recall (worn): {r_old*100:.1f}% @ old recall-first thr={old_thr:.2f}"
          f"  ->  {r_new*100:.1f}% @ new FP-first thr={thr:.3f}")
    _tradeoff_table(imgs)

    # ---- three-tier verdict (the shipped decision rule) ----
    tc_hi, tc_maybe = pick_tier_thresholds(imgs)
    tier_counts = _tier_table(imgs, tc_hi, tc_maybe)
    # placement/label health: positive SIDES whose grid-max still can't find a
    # module (blank) — these are the true misses (extreme angle / occlusion),
    # candidates for the exclusion list on the next round.
    pos_sides = [s for d in imgs.values() if d["label"] == 1 for s in d["probs"]]
    if pos_sides:
        frac_low = float(np.mean(np.array(pos_sides) < 0.2))
        print(f"  placement check: {frac_low*100:.1f}% of positive sides still "
              f"grid-max <0.2 (blank — extreme angle/occlusion)")
    # grid-max normal second-corner (what sets tc_hi) — must drop vs 0.976 baseline
    neg_in = [d for d in imgs.values() if d["label"] == 0 and d["dist"] == "real"
              and d["source"] != MEGLASS_SOURCE]
    if neg_in:
        sec = sorted((min(d["probs"]) for d in neg_in), reverse=True)[:5]
        print(f"  normal 2nd-corner top5 (sets tc_hi): {[round(x,3) for x in sec]}")

    # per-image report for error analysis
    report = {"threshold": thr, "old_recall_first_threshold": old_thr,
              "tc_hi": tc_hi, "tc_maybe": tc_maybe, "tier_counts": tier_counts,
              "policy": {"fp_budget_bulk": FP_BUDGET_BULK, "target_recall": TARGET_RECALL},
              "kfolds": KFOLDS, "metrics": report_metrics,
              "images": [{"image_id": k, "prob": round(d["prob"], 4),
                          "corner_probs": [round(x, 4) for x in d["probs"]],
                          "boxes_L": [round(x, 4) for x in d.get("boxes_L", [])],
                          "boxes_R": [round(x, 4) for x in d.get("boxes_R", [])],
                          "label": d["label"], "dist": d["dist"], "source": d["source"],
                          "path": d["path"],
                          "tier": tier_verdict(d["probs"], tc_hi, tc_maybe),
                          "pred": int(d["prob"] >= thr),
                          "error": (None if int(d["prob"] >= thr) == d["label"]
                                    else ("FP" if d["label"] == 0 else "FN"))}
                         for k, d in imgs.items()]}
    os.makedirs("models", exist_ok=True)
    with open(OOF_REPORT, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"  wrote {OOF_REPORT}")
    if DUMP_ERRORS:
        _dump_error_crops(items, imgs, tc_hi, tc_maybe)

    print("\ntraining final model on all data ...")
    m = train_one(items, seed=SEED * 1000 + KFOLDS)
    torch.save({"state_dict": m.state_dict(), "threshold": thr, "input": INPUT,
                "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                "aggregate": AGGREGATE, "corner_size": cfg.corner_size,
                "corner_yc": cfg.corner_yc, "corner_out": cfg.corner_out,
                # candidate grid (browser must scan the SAME boxes for parity)
                "corner_grid_sizes": list(cfg.corner_grid_sizes),
                "corner_grid_yc": list(cfg.corner_grid_yc),
                "corner_grid_out": list(cfg.corner_grid_out),
                # three-tier decision rule (the shipped verdict logic):
                # YES = both sides' grid-max >= tc_hi, MAYBE = one >= tc_maybe
                "tc_hi": tc_hi, "tc_maybe": tc_maybe}, MODEL_OUT)
    print(f"saved {MODEL_OUT} (tc_hi={tc_hi:.3f} tc_maybe={tc_maybe:.3f})")


def _dump_error_crops(items, imgs, tc_hi, tc_maybe):
    """Save the CENTER corner crops of every misclassified IMAGE (by tier verdict)
    to debug/corner_errors/. Ray-Ban wrong = NO; normal wrong = YES or MAYBE."""
    root = os.path.join("debug", "corner_errors")
    for sub in ("FP", "FN"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    bad = {}
    for k, d in imgs.items():
        v = tier_verdict(d["probs"], tc_hi, tc_maybe)
        wrong = (d["label"] == 1 and v == "NO") or (d["label"] == 0 and v != "NO")
        if wrong:
            bad[k] = (d, "FN" if d["label"] == 1 else "FP", v)
    n = 0
    for it in items:
        if not it.is_center or it.image_id not in bad:
            continue
        d, kind, v = bad[it.image_id]
        base = os.path.splitext(os.path.basename(it.path))[0]
        Image.fromarray(it.rgb).save(os.path.join(
            root, kind, f"{v}__{d['probs'][0]:.2f}_{d['probs'][1]:.2f}__{base}__{it.side}.png"))
        n += 1
    print(f"  DUMP_ERRORS: wrote {n} center crops of misclassified images to {root}/")


if __name__ == "__main__":
    main()
