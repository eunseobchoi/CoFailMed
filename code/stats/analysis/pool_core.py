"""Calibration splits, encoder-family selection, and evaluation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from analysis.common import subject_ids

CALIBRATION_FRACTION = 0.5


@dataclass(frozen=True)
class UnitSplit:
    calibration_units: tuple[str, ...]
    evaluation_units: tuple[str, ...]
    calibration_columns: tuple[int, ...]
    evaluation_columns: tuple[int, ...]
    column_units: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationSummary:
    means: dict[str, float]
    standard_errors: dict[str, float]
    best_model: str
    quality_cutoff: float
    functional_candidates: tuple[str, ...]
    quality_eligible: tuple[str, ...]
    bundle_means: dict[str, float]
    bundle_standard_errors: dict[str, float]
    best_bundle: str
    functional_bundles: tuple[str, ...]
    quality_eligible_bundles: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationVectors:
    all_fail: np.ndarray
    recovered: np.ndarray
    anchor_fail: np.ndarray
    marginal_dice: np.ndarray


def split_units(task: str, case_ids: tuple[str, ...], seed: int) -> UnitSplit:
    column_units = subject_ids(task, case_ids) or case_ids
    units = tuple(sorted(set(column_units)))
    if len(units) < 4:
        raise ValueError(f"{task}: need at least four independent units")
    shuffled = np.asarray(units)[np.random.default_rng(seed).permutation(len(units))]
    cut = int(np.floor(len(units) * CALIBRATION_FRACTION))
    calibration = tuple(sorted(str(unit) for unit in shuffled[:cut]))
    evaluation = tuple(sorted(str(unit) for unit in shuffled[cut:]))
    calibration_set = set(calibration)
    cal_columns = tuple(i for i, unit in enumerate(column_units) if unit in calibration_set)
    eval_columns = tuple(i for i, unit in enumerate(column_units) if unit not in calibration_set)
    return UnitSplit(calibration, evaluation, cal_columns, eval_columns, column_units)


def unit_matrix(matrix: np.ndarray, units: tuple[str, ...]) -> np.ndarray:
    ordered = tuple(sorted(set(units)))
    unit_array = np.asarray(units)
    return np.stack(
        [np.mean(matrix[:, unit_array == unit], axis=1) for unit in ordered],
        axis=1,
    )


def calibration_summary(
    keys: tuple[str, ...], matrix: np.ndarray, floor: float
) -> CalibrationSummary:
    if matrix.shape[1] < 2:
        raise ValueError("calibration requires at least two independent units")
    means_array = np.mean(matrix, axis=1)
    ses_array = np.std(matrix, axis=1, ddof=1) / np.sqrt(matrix.shape[1])
    means = dict(zip(keys, means_array.astype(float), strict=True))
    ses = dict(zip(keys, ses_array.astype(float), strict=True))
    bundles = tuple(key.split("/", 1)[0] for key in keys)
    bundle_names = tuple(sorted(set(bundles)))
    bundle_array = np.asarray(bundles)
    bundle_matrix = np.stack(
        [np.mean(matrix[bundle_array == bundle], axis=0) for bundle in bundle_names]
    )
    bundle_mean_values = np.mean(bundle_matrix, axis=1)
    bundle_se_values = np.std(bundle_matrix, axis=1, ddof=1) / np.sqrt(matrix.shape[1])
    bundle_means = dict(zip(bundle_names, bundle_mean_values.astype(float), strict=True))
    bundle_ses = dict(zip(bundle_names, bundle_se_values.astype(float), strict=True))
    bundle_ranked = sorted(bundle_names, key=lambda key: (-bundle_means[key], key))
    best_bundle = bundle_ranked[0]
    cutoff = bundle_means[best_bundle] - bundle_ses[best_bundle]
    functional_bundles = tuple(key for key in bundle_ranked if bundle_means[key] >= floor)
    quality_bundles = tuple(
        key for key in bundle_ranked if key in functional_bundles and bundle_means[key] >= cutoff
    )
    ranked = sorted(keys, key=lambda key: (-means[key], key))
    functional = tuple(key for key in ranked if key.split("/", 1)[0] in functional_bundles)
    quality = tuple(key for key in ranked if key.split("/", 1)[0] in quality_bundles)
    return CalibrationSummary(
        means,
        ses,
        functional[0],
        cutoff,
        functional,
        quality,
        bundle_means,
        bundle_ses,
        best_bundle,
        functional_bundles,
        quality_bundles,
    )


def top_quality_pool(
    candidates: Iterable[str], means: dict[str, float], size: int
) -> tuple[str, ...]:
    ranked = sorted(candidates, key=lambda key: (-means[key], key))
    if len(ranked) < size:
        raise ValueError(f"need {size} candidates, found {len(ranked)}")
    return tuple(ranked[:size])


def architecture_family(candidate: str) -> str:
    bundle = candidate.split("/", 1)[0]
    if bundle == "biomedclip":
        return bundle
    prefixes = (
        "resnet",
        "clip_",
        "dinov2_",
        "convnext_",
        "efficientnet_",
        "deit_",
        "mae_",
    )
    for prefix in prefixes:
        if bundle.startswith(prefix):
            return prefix.rstrip("_")
    return bundle


def architecture_pool(
    candidates: tuple[str, ...], means: dict[str, float], size: int
) -> tuple[str, ...]:
    winners: dict[str, str] = {}
    for key in sorted(candidates, key=lambda item: (-means[item], item)):
        winners.setdefault(architecture_family(key), key)
    return top_quality_pool(tuple(winners.values()), means, size)


def evaluation_vectors(
    keys: tuple[str, ...],
    matrix: np.ndarray,
    pools: tuple[tuple[str, ...], ...],
    anchor: str,
    threshold: float,
) -> EvaluationVectors:
    index = {key: position for position, key in enumerate(keys)}
    anchor_fail = matrix[index[anchor]] < threshold
    all_fail_rows, recovered_rows, marginal_rows = [], [], []
    for pool in pools:
        pool_values = matrix[[index[key] for key in pool]]
        failures = pool_values < threshold
        all_fail_rows.append(np.all(failures, axis=0))
        recovered_rows.append(anchor_fail & np.any(~failures, axis=0))
        marginal_rows.append(np.mean(pool_values, axis=0))
    return EvaluationVectors(
        np.mean(all_fail_rows, axis=0),
        np.mean(recovered_rows, axis=0),
        anchor_fail,
        np.mean(marginal_rows, axis=0),
    )
