"""Export the corner module CNN to ONNX for the browser, int8-quantize, and
validate onnxruntime vs PyTorch. Writes static/models/corner_clf.onnx (+ .meta.json
carrying the three-tier thresholds and corner-crop geometry).

    python tools/export_corner_onnx.py
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
from datasets_corners import load_corner_dataset
import train_corner_clf as TC

OUT_DIR = os.path.join("static", "models")
CKPT = "models/corner_clf.pt"
CALIB_MAX = 200


def _preprocess_batch(items):
    xs = [TC._eval_tf(Image.fromarray(it.rgb)).numpy() for it in items]
    return np.stack(xs).astype(np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    meta = {"input": ck["input"], "mean": ck["mean"], "std": ck["std"],
            "tc_hi": ck["tc_hi"], "tc_maybe": ck["tc_maybe"],
            "corner_size": ck["corner_size"], "corner_yc": ck["corner_yc"],
            "corner_out": ck["corner_out"]}

    TC.DEVICE = "cpu"
    m = TC.build_model().cpu()
    m.load_state_dict(ck["state_dict"])
    m.eval()

    dummy = torch.randn(2, 3, meta["input"], meta["input"])
    fp32 = os.path.join(tempfile.gettempdir(), "corner_fp32.onnx")
    torch.onnx.export(m, dummy, fp32, input_names=["input"], output_names=["logit"],
                      dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
                      opset_version=17, dynamo=False)
    print(f"exported fp32 onnx ({os.path.getsize(fp32)/1e6:.1f} MB)")

    items = load_corner_dataset(Config())
    arr = _preprocess_batch(items)
    calib = arr if len(arr) <= CALIB_MAX else arr[:: max(1, len(arr) // CALIB_MAX)]
    import onnxruntime as ort
    with torch.no_grad():
        torch_p = 1 / (1 + np.exp(-m(torch.from_numpy(arr)).numpy().ravel()))
    # verdict agreement at the tier thresholds (per corner)
    def validate(path, tag):
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        ort_p = 1 / (1 + np.exp(-sess.run(None, {"input": arr})[0].ravel()))
        d = float(np.max(np.abs(torch_p - ort_p)))
        agree = 0
        for thr in (meta["tc_hi"], meta["tc_maybe"]):
            agree = max(agree, int(((torch_p >= thr) != (ort_p >= thr)).sum()))
        print(f"  [{tag}] max|Δprob|={d:.4f}  worst tier-flips: {agree}/{len(arr)}")
        return agree, d

    print("validate fp32:")
    validate(fp32, "fp32")

    out = os.path.join(OUT_DIR, "corner_clf.onnx")
    from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                          QuantType, QuantFormat)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    prep = os.path.join(tempfile.gettempdir(), "corner_prep.onnx")
    quant_pre_process(fp32, prep)
    data = [{"input": calib[i:i + 1]} for i in range(len(calib))]

    class R(CalibrationDataReader):
        def __init__(self):
            self.i = 0

        def get_next(self):
            if self.i >= len(data):
                return None
            d = data[self.i]; self.i += 1
            return d

    quantize_static(prep, out, R(), quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8)
    print(f"int8 onnx ({os.path.getsize(out)/1e6:.2f} MB):")
    flips, _ = validate(out, "int8")

    if flips > 0.05 * len(arr):
        import shutil
        shutil.copy(fp32, out)
        print(f"  int8 degraded -> shipping fp32 ({os.path.getsize(out)/1e6:.1f} MB)")

    with open(os.path.join(OUT_DIR, "corner_clf.meta.json"), "w") as f:
        json.dump(meta, f)
    print("wrote corner_clf.meta.json:", meta)


if __name__ == "__main__":
    main()
