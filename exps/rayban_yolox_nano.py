#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""YOLOX-Nano experiment for the single-class Ray-Ban Meta glasses detector.

Trains a ~0.9M-param nano model on the exported COCO dataset at
data/rayban_yolox_coco/ (built by `tools/export_dataset.py --format yolox`).
The one class is `rayban_meta`; normal / no-glasses images are background
negatives (zero-annotation COCO images).

Run (from the repo root):
    .venv-train/bin/python third_party/YOLOX/tools/train.py \
        -f exps/rayban_yolox_nano.py -b 16 -c models/yolox_nano.pth

See MODEL_COMPARISON.md for how this compares against the YOLO11s baseline.
"""
import os

import torch.nn as nn

from yolox.exp import Exp as BaseExp

# repo root = parent of the exps/ directory holding this file
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "rayban_yolox"
        # runs/rayban_yolox/  (matches the existing runs/rayban_yolo layout)
        self.output_dir = os.path.join(_REPO, "runs")

        # --- nano architecture (depth/width + depthwise convs) ---
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.25

        # --- input resolution: strict fixed 640 (no multiscale). Base, training,
        #     eval, inference and ONNX export all run at exactly 640x640. ---
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.multiscale_range = 0            # disable multiscale range

        # --- augmentation: moderate. Offline aug (tools/augment_positives.py)
        #     already injects glare/occlusion/distance, so we keep mosaic light
        #     and mixup off (nano default) for this small, single-class set. ---
        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False
        self.degrees = 8.0
        self.translate = 0.1
        self.flip_prob = 0.5
        self.hsv_prob = 1.0

        # --- dataset (YOLOX COCODataset layout) ---
        self.data_dir = os.path.join(_REPO, "data", "rayban_yolox_coco")
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.data_num_workers = 4

        # --- schedule ---
        self.max_epoch = 150
        self.warmup_epochs = 3
        self.no_aug_epochs = 15
        self.eval_interval = 5
        self.print_interval = 20

    def get_model(self, sublinear=False):
        """Nano must build depthwise convs in both the FPN and the head — the
        base Exp builds regular convs, so this override is required (copied from
        exps/default/yolox_nano.py)."""
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        """Multiscale is disabled: always return the fixed base size so every
        training batch is exactly 640x640. (Also avoids the base implementation's
        CUDA-tensor broadcast, which would fail on the single-process Mac run.)"""
        return self.input_size
