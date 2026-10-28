"""Calibration thresholds and repeated-split means."""

from __future__ import annotations

from typing import Any

import numpy as np


def empirical_p33(
    keys: tuple[str, ...],
    calibration_cases: np.ndarray,
    anchor: str,
) -> float:
    """Return the calibration-best anchor's per-case P33 using calibration only."""
    index = {key: position for position, key in enumerate(keys)}
    values = np.asarray(calibration_cases[index[anchor]], dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("P33 requires non-empty finite calibration values")
    return float(np.percentile(values, 33.0))


def bootstrap_mean(
    values: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap the mean over finite repeated-split estimates."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"value": None, "ci95": None, "n_usable_splits": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, finite.size, size=(draws, finite.size))
    samples = np.mean(finite[indices], axis=1)
    return {
        "value": float(np.mean(finite)),
        "ci95": [float(value) for value in np.percentile(samples, [2.5, 97.5])],
        "n_usable_splits": int(finite.size),
    }
