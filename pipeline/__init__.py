"""Deterministic classical-CV pipeline for Meta smart-glasses detection.

No machine learning of any kind: only thresholding, contours, Hough circles,
distance transforms and hand-tuned geometric heuristics.

Stages:
    preprocess -> segment -> locate -> features -> decide -> (viz)
"""
