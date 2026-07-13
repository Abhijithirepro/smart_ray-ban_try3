"""Train the glasses-region CNN (Ray-Ban vs normal) by transfer learning.

MobileNetV3-small (ImageNet) fine-tuned on localized glasses-region crops
(pipeline/region). Reports grouped K-fold CV (out-of-fold, honest on this tiny
set), picks a decision threshold that keeps false positives low, then trains a
final model on ALL data and saves it for ONNX export.

    python train_region_clf.py            # CV + final model
Env knobs: EPOCHS_HEAD, EPOCHS_FT, KFOLDS, BATCH, INPUT.
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
from datasets import load_dataset

SEED = 0
INPUT = int(os.environ.get("INPUT", 160))
BATCH = int(os.environ.get("BATCH", 16))
EPOCHS_HEAD = int(os.environ.get("EPOCHS_HEAD", 12))
EPOCHS_FT = int(os.environ.get("EPOCHS_FT", 8))
KFOLDS = int(os.environ.get("KFOLDS", 4))
MODEL_OUT = "models/region_clf.pt"
OOF_REPORT = "models/oof_report.json"

# Threshold POLICY knobs (not tuned magic numbers — they state the operating point):
#   FP_BUDGET_BULK: fraction of MeGlass bulk negatives allowed above threshold (so one
#                   weird celebrity photo can't drag the threshold to 0.99).
#   TARGET_RECALL:  positives we insist on keeping when a class gap lets us.
FP_BUDGET_BULK = float(os.environ.get("FP_BUDGET_BULK", 0.01))
TARGET_RECALL = float(os.environ.get("TARGET_RECALL", 0.90))
DUMP_ERRORS = os.environ.get("DUMP_ERRORS") == "1"

# Bulk hard-negative tier (imported by tools/import_meglass.py). Everything else
# with label 0 is an IN-DOMAIN negative (the real deployment distribution).
MEGLASS_SOURCE = "data/normal/meglass"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

_train_tf = T.Compose([
    T.ToImage(),
    T.Resize((int(INPUT * 1.15), int(INPUT * 1.15))),
    T.RandomResizedCrop(INPUT, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
    T.RandomHorizontalFlip(),                 # Ray-Ban is L/R symmetric
    T.ColorJitter(0.3, 0.3, 0.3, 0.05),
    T.RandomRotation(10),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
_eval_tf = T.Compose([
    T.ToImage(),
    T.Resize((INPUT, INPUT)),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class CropDS(torch.utils.data.Dataset):
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
    """Sampling tier for balancing: positives, in-domain negatives, bulk (MeGlass)."""
    if it.label == 1:
        return "pos"
    return "bulk" if it.source == MEGLASS_SOURCE else "indomain"


def _make_sampler(items):
    """WeightedRandomSampler giving positives half the mass and negatives the other
    half, split evenly across whatever negative tiers are present. This keeps the 42
    in-domain worn normals in ~every other negative batch even when MeGlass floods
    the set, so the model can't take the 'webcam-look => Ray-Ban' shortcut. Balancing
    lives in the sampler, so the loss uses no pos_weight (that would double-correct)."""
    tiers = [_tier(it) for it in items]
    counts = {t: tiers.count(t) for t in set(tiers)}
    neg_present = [t for t in ("indomain", "bulk") if counts.get(t)]
    target = {"pos": 0.5} if counts.get("pos") else {}
    per_neg = (0.5 / len(neg_present)) if neg_present else 0.0
    for t in neg_present:
        target[t] = per_neg
    # renormalise in case positives are absent (degenerate) so weights sum to 1
    tot = sum(target.values()) or 1.0
    weights = [target[t] / tot / counts[t] for t in tiers]
    return torch.utils.data.WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double), num_samples=len(items), replacement=True)


def train_one(train_items):
    m = build_model()
    dl = torch.utils.data.DataLoader(CropDS(train_items, True), batch_size=BATCH,
                                     sampler=_make_sampler(train_items), num_workers=0)
    crit = nn.BCEWithLogitsLoss()   # sampler handles balance; no pos_weight

    # stage 1: frozen backbone, train head
    _set_backbone_grad(m, False)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(EPOCHS_HEAD):
        _epoch(m, dl, crit, opt)
    # stage 2: unfreeze, fine-tune at low LR
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
    dl = torch.utils.data.DataLoader(CropDS(items, False), batch_size=BATCH)
    out = []
    for x, _ in dl:
        p = torch.sigmoid(m(x.to(DEVICE))).cpu().numpy().ravel()
        out.extend(p.tolist())
    return np.array(out)


def grouped_folds(items, k):
    groups = {}
    for it in items:
        groups.setdefault(it.group, it.label)
    gl = list(groups.items())
    rng = np.random.RandomState(SEED)
    rng.shuffle(gl)
    # round-robin per class -> balanced folds
    fold_of = {}
    counters = {0: 0, 1: 0}
    for g, lab in gl:
        fold_of[g] = counters[lab] % k
        counters[lab] += 1
    return fold_of


def metrics(probs, labels, dists, thr):
    probs, labels = np.array(probs), np.array(labels)
    pred = (probs >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    npos = max(1, (labels == 1).sum()); nneg = max(1, (labels == 0).sum())
    return {"acc": (tp + tn) / len(labels), "recall": tp / npos,
            "fp_rate": fp / nneg, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _recall_first_threshold(probs, labels, target_recall=0.90):
    """The OLD policy, kept only to report how much recall the new threshold costs:
    highest threshold on a grid that still hits target_recall."""
    ts = np.linspace(0.05, 0.95, 37)
    best_t = None
    for t in ts:
        if metrics(probs, labels, None, t)["recall"] >= target_recall:
            best_t = t
    return float(best_t) if best_t is not None else float(ts[0])


def pick_threshold(probs, labels, dists, sources,
                   fp_budget_bulk=FP_BUDGET_BULK, target_recall=TARGET_RECALL):
    """FP-first threshold, placed by the observed out-of-fold distributions on the
    REAL (worn) subset — studio shots are trivially separable and only inflate the
    constraint, so they are excluded.

    Two negative tiers:
      * IN-DOMAIN (normal_glassess etc.) — the actual deployment distribution. The
        threshold must clear EVERY one of these => zero worn false positives by
        construction (Floor A).
      * BULK (MeGlass) — huge, slightly out-of-domain. Allow fp_budget_bulk of them
        above threshold so one odd photo can't force the threshold to ~1.0 (Floor B).

    Final threshold sits at the midpoint of the class gap when positives and the FP
    floor are separated (max robustness to drift on both sides); when they overlap it
    pins to the floor (false positives win, per the stated priority)."""
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
    return float(min(0.99, max(0.05, thr)))


def _fp_by_tier(probs, labels, sources, thr):
    """False-positive counts split into in-domain vs bulk (MeGlass) negatives."""
    probs, labels, sources = np.asarray(probs), np.asarray(labels), np.asarray(sources)
    fired = probs >= thr
    n_in = (labels == 0) & (sources != MEGLASS_SOURCE)
    n_bulk = (labels == 0) & (sources == MEGLASS_SOURCE)
    return {"indomain_fp": int((fired & n_in).sum()), "indomain_n": int(n_in.sum()),
            "bulk_fp": int((fired & n_bulk).sum()), "bulk_n": int(n_bulk.sum())}


def _dump_error_crops(items, oof, thr):
    """Save misclassified OOF crops to debug/oof_errors/{FP,FN}/ (worst first)."""
    root = os.path.join("debug", "oof_errors")
    for sub in ("FP", "FN"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    n = 0
    for it, p in zip(items, oof):
        if np.isnan(p):
            continue
        pred = int(p >= thr)
        if pred == it.label:
            continue
        kind = "FP" if it.label == 0 else "FN"
        base = os.path.splitext(os.path.basename(it.path))[0]
        name = f"{p:.3f}__{base}.png"
        Image.fromarray(it.rgb).save(os.path.join(root, kind, name))
        n += 1
    print(f"  DUMP_ERRORS: wrote {n} misclassified crops to {root}/")


def main():
    cfg = Config()
    print(f"device={DEVICE} loading dataset ...")
    items = load_dataset(cfg)
    labels = np.array([it.label for it in items])
    dists = np.array([it.dist for it in items])
    sources = np.array([it.source for it in items])
    n_bulk = int((sources == MEGLASS_SOURCE).sum())
    print(f"{len(items)} crops | rayban={(labels==1).sum()} normal={(labels==0).sum()} "
          f"(meglass bulk={n_bulk}) | groups={len({it.group for it in items})}")

    # ---- grouped K-fold CV (out-of-fold predictions) ----
    fold_of = grouped_folds(items, KFOLDS)
    oof = np.full(len(items), np.nan)
    fold_ids = np.full(len(items), -1)
    for k in range(KFOLDS):
        test_idx = [i for i, it in enumerate(items) if fold_of[it.group] == k]
        train_idx = [i for i in range(len(items)) if i not in set(test_idx)]
        if not test_idx:
            continue
        m = train_one([items[i] for i in train_idx])
        oof[test_idx] = predict(m, [items[i] for i in test_idx])
        for i in test_idx:
            fold_ids[i] = k
        print(f"  fold {k+1}/{KFOLDS}: trained on {len(train_idx)}, tested {len(test_idx)}")

    valid = ~np.isnan(oof)
    thr = pick_threshold(oof[valid], labels[valid], dists[valid], sources[valid])

    # per-fold thresholds: if these swing wildly the single number isn't trustworthy
    per_fold_thr = []
    for k in range(KFOLDS):
        idx = (fold_ids == k) & valid
        if idx.sum() and (labels[idx] == 1).any() and (labels[idx] == 0).any():
            per_fold_thr.append(round(pick_threshold(
                oof[idx], labels[idx], dists[idx], sources[idx]), 3))
    old_thr = _recall_first_threshold(oof[valid & (dists == "real")],
                                      labels[valid & (dists == "real")])

    print(f"\n=== GROUPED {KFOLDS}-FOLD CV (out-of-fold) | threshold={thr:.3f} ===")
    print(f"  policy: FP-first (fp_budget_bulk={FP_BUDGET_BULK}, target_recall={TARGET_RECALL})")
    print(f"  per-fold thresholds: {per_fold_thr}  (spread check)")
    overall = metrics(oof[valid], labels[valid], None, thr)
    print(f"  overall: acc={overall['acc']*100:.1f}%  recall(rayban)={overall['recall']*100:.1f}%  "
          f"FP-rate(normal)={overall['fp_rate']*100:.1f}%  [tp={overall['tp']} fp={overall['fp']} tn={overall['tn']} fn={overall['fn']}]")
    report_metrics = {"overall": overall}
    for d in ("studio", "real"):
        idx = valid & (dists == d)
        if idx.any():
            mm = metrics(oof[idx], labels[idx], None, thr)
            report_metrics[d] = mm
            print(f"  {d:6}: acc={mm['acc']*100:.1f}%  recall={mm['recall']*100:.1f}%  "
                  f"FP-rate={mm['fp_rate']*100:.1f}%  [tp={mm['tp']} fp={mm['fp']} tn={mm['tn']} fn={mm['fn']}]")
    # worn-subset FP by tier + recall cost vs the old recall-first policy
    real = valid & (dists == "real")
    tiers = _fp_by_tier(oof[real], labels[real], sources[real], thr)
    report_metrics["real_fp_by_tier"] = tiers
    print(f"  real FP by tier: in-domain {tiers['indomain_fp']}/{tiers['indomain_n']}  "
          f"bulk(meglass) {tiers['bulk_fp']}/{tiers['bulk_n']}")
    r_new = metrics(oof[real], labels[real], None, thr)["recall"]
    r_old = metrics(oof[real], labels[real], None, old_thr)["recall"]
    print(f"  recall cost (worn): {r_old*100:.1f}% @ old recall-first thr={old_thr:.2f}"
          f"  ->  {r_new*100:.1f}% @ new thr={thr:.3f}")

    # ---- OOF report (per-image, for error analysis + data collection) ----
    report = {
        "threshold": thr, "old_recall_first_threshold": old_thr,
        "policy": {"fp_budget_bulk": FP_BUDGET_BULK, "target_recall": TARGET_RECALL},
        "per_fold_thresholds": per_fold_thr, "kfolds": KFOLDS,
        "metrics": report_metrics,
        "items": [
            {"path": it.path, "label": int(it.label), "dist": it.dist,
             "source": it.source, "group": it.group,
             "prob": (None if np.isnan(oof[i]) else round(float(oof[i]), 4)),
             "pred": (None if np.isnan(oof[i]) else int(oof[i] >= thr)),
             "error": (None if np.isnan(oof[i]) or int(oof[i] >= thr) == it.label
                       else ("FP" if it.label == 0 else "FN"))}
            for i, it in enumerate(items)],
    }
    os.makedirs("models", exist_ok=True)
    with open(OOF_REPORT, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"  wrote {OOF_REPORT}")
    if DUMP_ERRORS:
        _dump_error_crops(items, oof, thr)

    # ---- final model on ALL data ----
    print("\ntraining final model on all data ...")
    m = train_one(items)
    torch.save({"state_dict": m.state_dict(), "threshold": thr, "input": INPUT,
                "mean": IMAGENET_MEAN, "std": IMAGENET_STD}, MODEL_OUT)
    print(f"saved {MODEL_OUT} (threshold={thr:.3f})")


if __name__ == "__main__":
    main()
