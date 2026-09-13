"""Shared distance helpers for SID construction."""

from __future__ import annotations

import numpy as np


DISTANCE_METRICS = ("euclidean", "cosine")


def validate_distance_metric(distance_metric: str) -> str:
    metric = str(distance_metric).lower()
    if metric not in DISTANCE_METRICS:
        raise ValueError(
            f"distance_metric must be one of {DISTANCE_METRICS}, got {metric!r}"
        )
    return metric


def normalize_numpy_rows(
    values: np.ndarray,
    *,
    copy: bool = True,
) -> np.ndarray:
    values = np.array(values, dtype=np.float32, copy=copy)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values /= np.maximum(norms, 1e-12)
    return values


def prepare_numpy_values(
    values: np.ndarray,
    distance_metric: str,
    *,
    copy: bool = True,
) -> np.ndarray:
    metric = validate_distance_metric(distance_metric)
    values = np.array(values, dtype=np.float32, copy=copy)
    if metric == "cosine":
        return normalize_numpy_rows(values, copy=False)
    return values
