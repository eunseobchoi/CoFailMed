#!/usr/bin/env python3
"""Validate and reproduce primary correlation statistics from per-case traces."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

TASKS = ("kvasir", "acdc_lv", "brats_wt", "riga_cup", "riga_disc")
FUNCTIONAL_FLOOR = 0.30
TOLERANCE = 3e-3
BOOTSTRAP_B = 0
RANDOM_SEED = 0
DINOV2_PAIR = frozenset({"dinov2_vitb14", "dinov2_vits14"})
PEARSON_STD_FLOOR = 1e-8
ARTIFACT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ARTIFACT_ROOT / "build/stats/trace_validation.json"


@dataclass(frozen=True)
class Trace:
    task: str
    bundle: str
    seed: int
    test_ids: tuple[str, ...]
    dice: np.ndarray
    source: str


def _validate_trace(trace: Trace) -> None:
    if trace.dice.ndim != 1:
        raise ValueError(f"{trace.source}: per_case_dice must be one-dimensional")
    if len(trace.test_ids) != len(trace.dice):
        raise ValueError(f"{trace.source}: test_ids and per_case_dice length mismatch")
    if len(set(trace.test_ids)) != len(trace.test_ids):
        raise ValueError(f"{trace.source}: duplicate test_ids")
    if not np.all(np.isfinite(trace.dice)):
        raise ValueError(f"{trace.source}: per_case_dice contains NaN or infinity")


def load_trace(path: Path, root: Path) -> Trace:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    trace = Trace(
        task=str(payload["task"]),
        bundle=str(payload.get("fm") or path.parent.name),
        seed=int(payload["seed"]),
        test_ids=tuple(str(case_id) for case_id in payload["test_ids"]),
        dice=np.asarray(payload["per_case_dice"], dtype=float),
        source=str(path.relative_to(root)),
    )
    _validate_trace(trace)
    return trace


def load_task_traces(
    root: Path,
    task: str,
    bundles: Iterable[str],
) -> list[Trace]:
    traces: list[Trace] = []
    for bundle in sorted(bundles):
        paths = sorted((root / "per_case_dice" / task / bundle).glob("seed_*.json"))
        if not paths:
            raise FileNotFoundError(f"{task}/{bundle}: no seed trace files")
        traces.extend(load_trace(path, root) for path in paths)
    if any(trace.task != task for trace in traces):
        wrong = next(trace for trace in traces if trace.task != task)
        raise ValueError(f"{wrong.source}: task is {wrong.task!r}, expected {task!r}")
    return traces


def align_pair(
    left: Trace, right: Trace, *, allow_partial: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Pair two traces by test_id; the ID sets must match unless ``allow_partial=True``."""
    left_map = dict(zip(left.test_ids, left.dice, strict=True))
    right_map = dict(zip(right.test_ids, right.dice, strict=True))
    if left_map.keys() != right_map.keys() and not allow_partial:
        raise ValueError(
            f"{left.source} vs {right.source}: test_ids differ; pass allow_partial=True to intersect"
        )
    common = sorted(left_map.keys() & right_map.keys())
    if len(common) < 2:
        raise ValueError(
            f"{left.source} vs {right.source}: fewer than two intersecting test_ids"
        )
    return (
        np.asarray([left_map[key] for key in common], dtype=float),
        np.asarray([right_map[key] for key in common], dtype=float),
    )


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"Pearson shape mismatch: {left_arr.shape} vs {right_arr.shape}")
    if left_arr.ndim != 1 or left_arr.size < 2:
        raise ValueError("Pearson requires at least two paired one-dimensional values")
    if not np.all(np.isfinite(left_arr)) or not np.all(np.isfinite(right_arr)):
        raise ValueError("Pearson input contains NaN or infinity")
    if (
        not pearson_input_has_variance(left_arr)
        or not pearson_input_has_variance(right_arr)
    ):
        raise ValueError("Pearson input has zero variance")
    return float(np.corrcoef(left_arr, right_arr)[0, 1])


def pearson_input_has_variance(values: np.ndarray) -> bool:
    """Return whether values satisfy the shared Pearson variance floor."""

    array = np.asarray(values, dtype=float)
    return bool(
        array.ndim == 1
        and array.size >= 2
        and np.all(np.isfinite(array))
        and np.std(array) >= PEARSON_STD_FLOOR
    )


def apply_bundle_floor(traces: list[Trace], floor: float) -> list[Trace]:
    grouped: dict[str, list[Trace]] = {}
    for trace in traces:
        grouped.setdefault(trace.bundle, []).append(trace)
    means = {
        bundle: float(np.mean([float(np.mean(trace.dice)) for trace in members]))
        for bundle, members in grouped.items()
    }
    return [trace for trace in traces if means[trace.bundle] >= floor]


def within_cross(
    traces: list[Trace],
    same_bundle_pairs: set[frozenset[str]],
) -> dict[str, float | int]:
    same_values: list[float] = []
    cross_values: list[float] = []
    ordered = sorted(traces, key=lambda item: (item.bundle, item.seed))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            is_same = (
                left.bundle == right.bundle
                or frozenset({left.bundle, right.bundle}) in same_bundle_pairs
            )
            aligned_left, aligned_right = align_pair(left, right)
            value = pearson(aligned_left, aligned_right)
            (same_values if is_same else cross_values).append(value)
    if not same_values or not cross_values:
        raise ValueError("within/cross computation produced an empty stratum")
    within_mean = float(np.mean(same_values))
    cross_mean = float(np.mean(cross_values))
    return {
        "within_rbar": within_mean,
        "cross_rbar": cross_mean,
        "gap": within_mean - cross_mean,
        "n_within_pairs": len(same_values),
        "n_cross_pairs": len(cross_values),
    }


def _comparison(calculated: Any, expected: Any, tolerance: float) -> dict[str, Any]:
    difference = (
        int(calculated) - int(expected)
        if isinstance(expected, int)
        else float(calculated) - float(expected)
    )
    passed = difference == 0 if isinstance(expected, int) else abs(difference) <= tolerance
    return {
        "calculated": calculated,
        "expected": expected,
        "difference": difference,
        "status": "PASS" if passed else "FAIL",
    }


def reproduce_task(root: Path, task: str) -> dict[str, Any]:
    expected_path = root / "within_cross" / f"{task}.json"
    with expected_path.open("r", encoding="utf-8") as handle:
        expected_payload = json.load(handle)
    available = tuple(expected_payload["fms_available_before_floor"])
    all_traces = load_task_traces(root, task, available)
    kept_traces = apply_bundle_floor(all_traces, FUNCTIONAL_FLOOR)
    calculated = within_cross(kept_traces, {DINOV2_PAIR})
    keys = (
        "within_rbar",
        "cross_rbar",
        "gap",
        "n_within_pairs",
        "n_cross_pairs",
    )
    comparisons = {
        key: _comparison(calculated[key], expected_payload[key], TOLERANCE)
        for key in keys
    }
    kept = sorted({trace.bundle for trace in kept_traces})
    floor_match = kept == sorted(expected_payload["fms_kept_after_floor"])
    passed = floor_match and all(row["status"] == "PASS" for row in comparisons.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "comparisons": comparisons,
        "floor": {
            "calculated": kept,
            "expected": sorted(expected_payload["fms_kept_after_floor"]),
            "status": "PASS" if floor_match else "FAIL",
        },
        "n_bundles": len(kept),
        "n_cases": len(kept_traces[0].test_ids),
        "source_files": [trace.source for trace in all_traces]
        + [str(expected_path.relative_to(root))],
    }


def build_payload(root: Path) -> dict[str, Any]:
    tasks = {task: reproduce_task(root, task) for task in TASKS}
    source_files = sorted(
        {source for row in tasks.values() for source in row["source_files"]}
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PASS" if all(row["status"] == "PASS" for row in tasks.values()) else "FAIL"
        ),
        "tolerance": TOLERANCE,
        "functional_floor": FUNCTIONAL_FLOOR,
        "n_bundles": {task: row["n_bundles"] for task, row in tasks.items()},
        "n_cases": {task: row["n_cases"] for task, row in tasks.items()},
        "bootstrap_B": BOOTSTRAP_B,
        "random_seed": RANDOM_SEED,
        "source_files": source_files,
        "tasks": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_payload(args.results_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"{payload['status']}: {args.output}")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
