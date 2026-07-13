"""Export the trained region CNN to ONNX, int8-quantize it for the browser, and
validate onnxruntime vs PyTorch. Writes static/models/region_clf.onnx (+ .meta.json).

    python tools/export_onnx.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from datasets import load_dataset
import train_region_clf as TR

OUT_DIR = os.path.join("static", "models")
CKPT = "models/region_clf.pt"


def _preprocess_batch(items, meta):
    """Eval transform -> NCHW float32 numpy, matching training/inference exactly."""
    xs = []
    for it in items:
        t = TR._eval_tf(Image.fromarray(it.rgb))  # CHW float tensor, normalized
        xs.append(t.numpy())
    return np.stack(xs).astype(np.float32)


class _CalibReader:
    def __init__(self, arr, input_name):
        self.data = [{input_name: arr[i:i + 1]} for i in range(len(arr))]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.data):
            return None
        d = self.data[self.i]; self.i += 1
        return d


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    meta = {"input": ck["input"], "mean": ck["mean"], "std": ck["std"],
            "threshold": ck["threshold"]}

    # rebuild model on CPU
    TR.DEVICE = "cpu"
    m = TR.build_model().cpu()
    m.load_state_dict(ck["state_dict"])
    m.eval()

    dummy = torch.randn(2, 3, meta["input"], meta["input"])
    fp32 = os.path.join(tempfile.gettempdir(), "region_fp32.onnx")
    # legacy (TorchScript) exporter — rounds-trips cleanly through ORT quantization
    torch.onnx.export(m, dummy, fp32, input_names=["input"], output_names=["logit"],
                      dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
                      opset_version=17, dynamo=False)
    print(f"exported fp32 onnx ({os.path.getsize(fp32)/1e6:.1f} MB)")

    items = load_dataset(Config())
    arr = _preprocess_batch(items, meta)            # ALL items -> validation
    thr = meta["threshold"]

    # int8 calibration only needs a representative sample; cap it so export stays
    # fast now that MeGlass makes the set ~600+ images. Stride-sample keeps the
    # class/source mix roughly intact without sorting.
    CALIB_MAX = 200
    calib_arr = arr if len(arr) <= CALIB_MAX else arr[:: max(1, len(arr) // CALIB_MAX)]
    import onnxruntime as ort
    with torch.no_grad():
        torch_p = (1 / (1 + np.exp(-m(torch.from_numpy(arr)).numpy().ravel())))

    def validate(path, tag):
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        ort_p = (1 / (1 + np.exp(-sess.run(None, {"input": arr})[0].ravel())))
        agree = int(((torch_p >= thr) == (ort_p >= thr)).sum())
        d = float(np.max(np.abs(torch_p - ort_p)))
        print(f"  [{tag}] max|Δprob|={d:.4f}  verdict agree @thr={thr:.2f}: {agree}/{len(arr)}")
        return agree, d

    print("validate fp32:")
    validate(fp32, "fp32")

    out = os.path.join(OUT_DIR, "region_clf.onnx")
    from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                          QuantType, QuantFormat)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    prep = os.path.join(tempfile.gettempdir(), "region_prep.onnx")
    quant_pre_process(fp32, prep)                    # fixes the version-converter issues
    reader = _CalibReader(calib_arr, "input")

    class R(CalibrationDataReader):
        def get_next(self_inner):
            return reader.get_next()

    quantize_static(prep, out, R(), quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8)
    print(f"int8 onnx ({os.path.getsize(out)/1e6:.2f} MB):")
    agree_i, _ = validate(out, "int8")

    # ship int8 if it round-trips well, else fall back to fp32
    if agree_i < 0.95 * len(arr):
        import shutil
        shutil.copy(fp32, out)
        print(f"  int8 degraded -> shipping fp32 ({os.path.getsize(out)/1e6:.1f} MB)")

    with open(os.path.join(OUT_DIR, "region_clf.meta.json"), "w") as f:
        json.dump(meta, f)
    print("wrote region_clf.meta.json:", meta)


if __name__ == "__main__":
    main()
