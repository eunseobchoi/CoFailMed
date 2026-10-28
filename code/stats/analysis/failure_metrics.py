#!/usr/bin/env python3
"""Binary failure metrics for the primary same/release and cross-bundle groups."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from analysis.common import (
    BOOTSTRAP_B,
    OUTPUT_ROOT,
    RANDOM_SEED,
    TASKS,
    aggregate_columns,
    aligned_matrix,
    now,
    primary_traces,
    source_files,
    subject_ids,
    write_json,
)
from analysis.trace_validation import DINOV2_PAIR

DIRECTIONS = {
    "kappa": "higher means more chance-adjusted dependence",
    "q_statistic": "higher means more dependence; -1 is inverse and +1 concordant",
    "disagreement_rate": "higher means more diversity and therefore less dependence",
    "double_fault_rate": "higher means more joint failure; not chance-adjusted",
}
# Error-consistency kappa when c_obs == 1 (e.g. both members never fail, so c_exp == 1 and the ratio is 0/0):
# "official" scores kappa = 1 as wichmann-lab/error-consistency@eb53bdb data-analysis-helper.R:242-248 does;
# "exclude" leaves it undefined (NaN) so the pair is dropped from the stratum mean.
KAPPA_NEVER_FAIL = ("official", "exclude")


def _check_kappa_never_fail(kappa_never_fail: str) -> None:
    if kappa_never_fail not in KAPPA_NEVER_FAIL:
        raise ValueError(f"kappa_never_fail must be one of {KAPPA_NEVER_FAIL}, got {kappa_never_fail!r}")


def _binary_inputs(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_arr = np.asarray(left, dtype=bool)
    right_arr = np.asarray(right, dtype=bool)
    if left_arr.shape != right_arr.shape or left_arr.ndim != 1 or left_arr.size == 0:
        raise ValueError("binary metric inputs must be non-empty matched vectors")
    return left_arr, right_arr


def _counts(left: np.ndarray, right: np.ndarray) -> tuple[int, int, int, int]:
    left_arr, right_arr = _binary_inputs(left, right)
    n11 = int(np.sum(left_arr & right_arr))
    n00 = int(np.sum(~left_arr & ~right_arr))
    n01 = int(np.sum(~left_arr & right_arr))
    n10 = int(np.sum(left_arr & ~right_arr))
    return n11, n00, n01, n10


def cohen_kappa(
    left: np.ndarray, right: np.ndarray, kappa_never_fail: str = "official"
) -> float | None:
    _check_kappa_never_fail(kappa_never_fail)
    left_arr, right_arr = _binary_inputs(left, right)
    observed = float(np.mean(left_arr == right_arr))
    p_left = float(np.mean(left_arr))
    p_right = float(np.mean(right_arr))
    expected = p_left * p_right + (1.0 - p_left) * (1.0 - p_right)
    denominator = 1.0 - expected
    if kappa_never_fail == "official" and observed == 1.0:
        return 1.0
    return None if denominator == 0.0 else (observed - expected) / denominator


def q_statistic(left: np.ndarray, right: np.ndarray) -> float | None:
    n11, n00, n01, n10 = _counts(left, right)
    numerator = n11 * n00 - n01 * n10
    denominator = n11 * n00 + n01 * n10
    return None if denominator == 0 else float(numerator / denominator)


def disagreement_rate(left: np.ndarray, right: np.ndarray) -> float:
    n11, n00, n01, n10 = _counts(left, right)
    return float((n01 + n10) / (n11 + n00 + n01 + n10))


def double_fault_rate(left: np.ndarray, right: np.ndarray) -> float:
    n11, n00, n01, n10 = _counts(left, right)
    return float(n00 / (n11 + n00 + n01 + n10))


def empirical_threshold(values: np.ndarray, percentile: float) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("empirical threshold input is empty or non-finite")
    return float(np.percentile(array, percentile))


def _same_mask(keys: list[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_i, pair_j = np.triu_indices(len(keys), k=1)
    same = np.asarray(
        [
            keys[i][0] == keys[j][0]
            or frozenset({keys[i][0], keys[j][0]}) == DINOV2_PAIR
            for i, j in zip(pair_i, pair_j, strict=True)
        ],
        dtype=bool,
    )
    return pair_i, pair_j, same


def _metric_matrices(success: np.ndarray, kappa_never_fail: str = "official") -> dict[str, np.ndarray]:
    _check_kappa_never_fail(kappa_never_fail)
    data = np.asarray(success, dtype=float)
    total = float(data.shape[1])
    n11 = data @ data.T
    success_count = np.sum(data, axis=1)
    n10 = success_count[:, None] - n11
    n01 = success_count[None, :] - n11
    n00 = total - n11 - n01 - n10
    observed = (n11 + n00) / total
    p = data.mean(axis=1)
    expected = np.outer(p, p) + np.outer(1.0 - p, 1.0 - p)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = (observed - expected) / (1.0 - expected)
        q_value = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10)
    if kappa_never_fail == "official":
        kappa = np.where(observed == 1.0, 1.0, kappa)
    return {
        "kappa": kappa,
        "q_statistic": q_value,
        "disagreement_rate": (n01 + n10) / total,
        "double_fault_rate": n00 / total,
    }


def _stratum(values: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    selected = np.asarray(values[mask], dtype=float)
    finite = selected[np.isfinite(selected)]
    if finite.size == 0:
        return float("nan"), 0
    return float(np.mean(finite)), int(finite.size)


def metric_summary(
    keys: list[tuple[str, int]],
    success: np.ndarray,
    kappa_never_fail: str = "official",
) -> dict[str, dict[str, float | int]]:
    pair_i, pair_j, same_mask = _same_mask(keys)
    output: dict[str, dict[str, float | int]] = {}
    for name, matrix in _metric_matrices(success, kappa_never_fail).items():
        values = matrix[pair_i, pair_j]
        same, n_same = _stratum(values, same_mask)
        cross, n_cross = _stratum(values, ~same_mask)
        output[name] = {
            "same": same,
            "cross": cross,
            "gap": same - cross,
            "n_valid_same_pairs": n_same,
            "n_valid_cross_pairs": n_cross,
        }
    return output


def _ci(values: list[float]) -> list[float]:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if finite.size == 0:
        return [float("nan"), float("nan")]
    return [float(value) for value in np.percentile(finite, [2.5, 97.5])]


def _bootstrap(
    keys: list[tuple[str, int]],
    success: np.ndarray,
    n_boot: int,
    seed: int,
    kappa_never_fail: str = "official",
) -> dict[str, dict[str, list[float]]]:
    rng = np.random.default_rng(seed)
    draws = {
        metric: {stratum: [] for stratum in ("same", "cross", "gap")}
        for metric in DIRECTIONS
    }
    n_cases = success.shape[1]
    for _ in range(n_boot):
        indices = rng.integers(0, n_cases, n_cases)
        summary = metric_summary(keys, success[:, indices], kappa_never_fail)
        for metric in DIRECTIONS:
            for stratum in ("same", "cross", "gap"):
                draws[metric][stratum].append(float(summary[metric][stratum]))
    return {
        metric: {stratum: _ci(values) for stratum, values in strata.items()}
        for metric, strata in draws.items()
    }


def _threshold_values(matrix: np.ndarray) -> dict[str, float]:
    return {
        "empirical_p33": empirical_threshold(matrix, 33.0),
        "dice_lt_0_5": 0.5,
        "dice_lt_0_7": 0.7,
    }


def _analyze_matrix(
    keys: list[tuple[str, int]],
    matrix: np.ndarray,
    n_boot: int,
    seed: int,
    kappa_never_fail: str = "official",
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for offset, (label, threshold) in enumerate(_threshold_values(matrix).items()):
        success = matrix >= threshold
        point = metric_summary(keys, success, kappa_never_fail)
        intervals = _bootstrap(keys, success, n_boot, seed + offset * 1009, kappa_never_fail)
        output[label] = {
            "failure_rule": f"Dice < {threshold:.17g}",
            "threshold": threshold,
            "metrics": {
                metric: {
                    "direction": DIRECTIONS[metric],
                    **point[metric],
                    "case_bootstrap_95ci": intervals[metric],
                }
                for metric in DIRECTIONS
            },
        }
    return output


def analyze_task(
    root: Path, task: str, n_boot: int, seed: int, kappa_never_fail: str = "official"
) -> dict[str, Any]:
    traces, expected_file = primary_traces(root, task)
    keys, matrix, case_ids = aligned_matrix(traces)
    units = subject_ids(task, case_ids)
    subject_result = None
    if units is not None:
        aggregated, unit_names = aggregate_columns(matrix, units)
        subject_result = {
            "n_units": len(unit_names),
            "thresholds": _analyze_matrix(keys, aggregated, n_boot, seed + 5003, kappa_never_fail),
        }
    return {
        "n_bundles": len({key[0] for key in keys}),
        "n_cases": len(case_ids),
        "n_members": len(keys),
        "fms_kept_after_floor": sorted({key[0] for key in keys}),
        "threshold_definition": (
            "P33 is the poolwide empirical percentile over all post-floor "
            "model-case Dice values and is fixed during bootstrap"
        ),
        "thresholds": _analyze_matrix(keys, matrix, n_boot, seed, kappa_never_fail),
        "subject_aggregated": subject_result,
        "source_files": source_files(traces, [expected_file]),
    }


def build_payload(
    root: Path, n_boot: int, seed: int, kappa_never_fail: str = "official"
) -> dict[str, Any]:
    tasks = {
        task: analyze_task(root, task, n_boot, seed + index * 10007, kappa_never_fail)
        for index, task in enumerate(TASKS)
    }
    return {
        "generated_at": now(),
        "n_bundles": {task: row["n_bundles"] for task, row in tasks.items()},
        "n_cases": {task: row["n_cases"] for task, row in tasks.items()},
        "bootstrap_B": n_boot,
        "kappa_never_fail": kappa_never_fail,
        "random_seed": seed,
        "source_files": sorted(
            {path for row in tasks.values() for path in row["source_files"]}
        ),
        "tasks": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "L1A_alternative_diversity.json")
    parser.add_argument("--bootstrap-b", type=int, default=BOOTSTRAP_B)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--kappa-never-fail", choices=KAPPA_NEVER_FAIL, default="official")
    args = parser.parse_args()
    write_json(
        args.output,
        build_payload(args.results_root, args.bootstrap_b, args.seed, args.kappa_never_fail),
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
