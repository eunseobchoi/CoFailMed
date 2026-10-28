"""Failure-event metrics and Pearson-ordering checks for the UNet-skip and nnU-Net controls."""
from __future__ import annotations
import re
from typing import Any
import numpy as np
from analysis.common import FUNCTIONAL_FLOOR
from analysis.effective_size import MIN_FAILURES, MIN_PASSES, TraceCollection
from analysis.failure_metrics import DIRECTIONS, _metric_matrices

METRICS = tuple(DIRECTIONS)
DEPENDENCE_DIRECTION = {
    "kappa": 1,
    "q_statistic": 1,
    "disagreement_rate": -1,
    "double_fault_rate": 1,
}
UNDEFINED_REASON = "undefined — zero/near-zero indicator variance"


def _pair_arrays(
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    array = np.asarray(pairs, dtype=int)
    if array.ndim != 2 or array.shape[1:] != (2,) or len(array) == 0:
        raise ValueError("pair stratum must contain at least one unique pair")
    return array


def nnunet_pair_masks(member_ids: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    folds: list[int] = []
    for member in member_ids:
        match = re.fullmatch(r"fold([0-4])_seed(?:13|37)", member)
        if match is None:
            raise ValueError(f"unexpected case-identical nnU-Net member {member!r}")
        folds.append(int(match.group(1)))
    within: list[tuple[int, int]] = []
    cross: list[tuple[int, int]] = []
    for left in range(len(member_ids)):
        for right in range(left + 1, len(member_ids)):
            (within if folds[left] == folds[right] else cross).append((left, right))
    return _pair_arrays(within), _pair_arrays(cross)


def _pearson_point(
    matrix: np.ndarray,
    within_pairs: np.ndarray,
    cross_pairs: np.ndarray,
) -> dict[str, float | int]:
    correlations = np.corrcoef(matrix)
    if not np.all(np.isfinite(correlations)):
        raise ValueError("Pearson input contains a degenerate member")
    within = correlations[within_pairs[:, 0], within_pairs[:, 1]]
    cross = correlations[cross_pairs[:, 0], cross_pairs[:, 1]]
    same = float(np.mean(within))
    different = float(np.mean(cross))
    return {
        "same": same,
        "cross": different,
        "gap_same_minus_cross": same - different,
        "n_same_pairs": int(len(within_pairs)),
        "n_cross_pairs": int(len(cross_pairs)),
    }


def _support(
    matrix: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    failures = np.sum(matrix < threshold, axis=1).astype(int)
    passes = (matrix.shape[1] - failures).astype(int)
    valid = (failures >= MIN_FAILURES) & (passes >= MIN_PASSES)
    return {
        "status": "support-valid" if np.all(valid) else UNDEFINED_REASON,
        "min_failures_required": MIN_FAILURES,
        "min_passes_required": MIN_PASSES,
        "member_failures": failures.tolist(),
        "member_passes": passes.tolist(),
        "member_support_valid": valid.tolist(),
        "n_members_supported": int(np.sum(valid)),
        "n_members_total": int(len(valid)),
    }


def _floor_gate(collection: TraceCollection) -> dict[str, Any]:
    bundle_means = {
        bundle: float(
            np.mean(
                collection.matrix[
                    np.asarray(collection.bundle_ids, dtype=object) == bundle
                ]
            )
        )
        for bundle in sorted(set(collection.bundle_ids))
    }
    below_floor = [
        bundle
        for bundle, mean_dice in bundle_means.items()
        if mean_dice < FUNCTIONAL_FLOOR
    ]
    if below_floor:
        raise RuntimeError(
            f"{collection.config}/{collection.task}: fixed companion members "
            f"fail Dice floor {FUNCTIONAL_FLOOR}: {below_floor}"
        )
    return {
        "status": "PASS",
        "functional_floor_mean_dice": FUNCTIONAL_FLOOR,
        "bundle_mean_dice": bundle_means,
        "bundles_below_floor": [],
        "fixed_members_retained_without_reselection": True,
        "evidence_tag": "MEASURED",
    }


def _stratum(
    values: np.ndarray,
    pairs: np.ndarray,
) -> tuple[float | None, int]:
    selected = np.asarray(values[pairs[:, 0], pairs[:, 1]], dtype=float)
    finite = selected[np.isfinite(selected)]
    return (
        (None, 0)
        if finite.size == 0
        else (float(np.mean(finite)), int(finite.size))
    )


def metric_points(
    success: np.ndarray,
    within_pairs: np.ndarray,
    cross_pairs: np.ndarray,
) -> dict[str, dict[str, Any]]:
    # The recorded control scorer reports undefined kappa cells, so it keeps the exclusion convention.
    matrices = _metric_matrices(success, kappa_never_fail="exclude")
    output: dict[str, dict[str, Any]] = {}
    for metric, matrix in matrices.items():
        same, n_same = _stratum(matrix, within_pairs)
        cross, n_cross = _stratum(matrix, cross_pairs)
        gap = None if same is None or cross is None else same - cross
        output[metric] = {
            "same": same,
            "cross": cross,
            "gap_same_minus_cross": gap,
            "n_valid_same_pairs": n_same,
            "n_total_same_pairs": int(len(within_pairs)),
            "n_undefined_same_pairs": int(len(within_pairs) - n_same),
            "n_valid_cross_pairs": n_cross,
            "n_total_cross_pairs": int(len(cross_pairs)),
            "n_undefined_cross_pairs": int(len(cross_pairs) - n_cross),
        }
    return output


def _ci(values: list[float]) -> list[float] | None:
    if not values:
        return None
    return [
        float(value)
        for value in np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
    ]


def bootstrap_metrics(
    matrix: np.ndarray,
    threshold: float,
    within_pairs: np.ndarray,
    cross_pairs: np.ndarray,
    *,
    b: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    draws = {
        metric: {key: [] for key in ("same", "cross", "gap_same_minus_cross")}
        for metric in METRICS
    }
    support_valid_draws = 0
    rng = np.random.default_rng(seed)
    for _ in range(b):
        indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        sampled = matrix[:, indices]
        if _support(sampled, threshold)["status"] != "support-valid":
            continue
        support_valid_draws += 1
        point = metric_points(
            sampled >= threshold,
            within_pairs,
            cross_pairs,
        )
        for metric in METRICS:
            for key in draws[metric]:
                value = point[metric][key]
                if value is not None and np.isfinite(value):
                    draws[metric][key].append(float(value))
    return {
        metric: {
            key: {
                "ci95": _ci(values),
                "B_finite": len(values),
            }
            for key, values in metric_draws.items()
        }
        | {
            "B_requested": b,
            "B_support_valid": support_valid_draws,
        }
        for metric, metric_draws in draws.items()
    }


def classify_order(
    metric: str,
    metric_gap: float | None,
    pearson_gap: float,
    ci95: list[float] | None,
    *,
    complete_pair_coverage: bool = True,
) -> dict[str, Any]:
    if metric_gap is None or not np.isfinite(metric_gap):
        return {
            "status": "undefined",
            "preserves_pearson_point_ordering": False,
            "gap_ci_excludes_zero": False,
            "evidence_tag": "DERIVED",
        }
    if not complete_pair_coverage:
        return {
            "status": "incomplete_pair_coverage",
            "preserves_pearson_point_ordering": False,
            "gap_ci_excludes_zero": bool(
                ci95 is not None and (ci95[1] < 0.0 or ci95[0] > 0.0)
            ),
            "evidence_tag": "DERIVED",
        }
    expected_sign = int(np.sign(pearson_gap)) * DEPENDENCE_DIRECTION[metric]
    observed_sign = int(np.sign(metric_gap))
    preserves = expected_sign != 0 and observed_sign == expected_sign
    excludes_zero = bool(
        ci95 is not None and (ci95[1] < 0.0 or ci95[0] > 0.0)
    )
    return {
        "status": "preserved" if preserves else "discordant",
        "pearson_gap_sign": int(np.sign(pearson_gap)),
        "metric_dependence_direction": DEPENDENCE_DIRECTION[metric],
        "expected_raw_gap_sign": expected_sign,
        "observed_raw_gap_sign": observed_sign,
        "preserves_pearson_point_ordering": preserves,
        "gap_ci_excludes_zero": excludes_zero,
        "evidence_tag": "DERIVED",
    }
