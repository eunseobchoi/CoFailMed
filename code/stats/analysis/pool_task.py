"""Repeated calibration/evaluation splits for encoder-family pools."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from analysis.common import (
    BOOTSTRAP_B,
    FUNCTIONAL_FLOOR,
    aligned_matrix,
    collect_traces,
    now,
    source_files,
)
from analysis.pool_config import (
    METRIC_NAMES,
    POOL_SIZE,
    PRIMARY_TASKS,
    RULE_DEFINITION,
    RULE_NAMES,
    SPLIT_COUNT,
    TASKS,
    THRESHOLD_JUSTIFICATION,
)
from analysis.pool_core import (
    CalibrationSummary,
    architecture_pool,
    calibration_summary,
    evaluation_vectors,
    split_units,
    unit_matrix,
)
from analysis.pool_evaluation import point
from analysis.pool_repeated import (
    bootstrap_mean,
    empirical_p33,
)


def _load_task(
    root: Path, task: str
) -> tuple[tuple[str, ...], np.ndarray, tuple[str, ...], list[Any], list[str]]:
    subdir = "per_case_dice" if task in PRIMARY_TASKS else "per_case_dice_heldout_11fm"
    traces = collect_traces(root, subdir, task)
    extras: list[str] = []
    if task in PRIMARY_TASKS:
        expected_path = root / "within_cross" / f"{task}.json"
        with expected_path.open("r", encoding="utf-8") as handle:
            available = set(json.load(handle)["fms_available_before_floor"])
        traces = [trace for trace in traces if trace.bundle in available]
        extras.append(str(expected_path.relative_to(root)))
    trace_keys, matrix, case_ids = aligned_matrix(traces)
    keys = tuple(f"{bundle}/seed_{seed}" for bundle, seed in trace_keys)
    return keys, matrix, case_ids, traces, extras


def _selections(
    summary: CalibrationSummary,
) -> dict[str, tuple[tuple[str, ...], ...]]:
    return {
        "architecture_diversity": (
            architecture_pool(summary.functional_candidates, summary.means, POOL_SIZE),
        ),
    }


def _evaluate_split_threshold(
    keys: tuple[str, ...],
    evaluation: np.ndarray,
    anchor: str,
    threshold: float,
    selections: dict[str, tuple[tuple[str, ...], ...]],
) -> dict[str, Any]:
    vectors = {
        rule: evaluation_vectors(keys, evaluation, pools, anchor, threshold)
        for rule, pools in selections.items()
    }
    metrics = {rule: point(value) for rule, value in vectors.items()}
    anchor_failures = int(np.sum(next(iter(vectors.values())).anchor_fail))
    return {
        "rules": metrics,
        "n_evaluation_cases": int(evaluation.shape[1]),
        "n_anchor_failure_cases": anchor_failures,
        "all_fail_case_events_by_rule": {
            rule: float(np.sum(value.all_fail)) for rule, value in vectors.items()
        },
    }


def _aggregate_threshold(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    rules: dict[str, Any] = {}
    for rule in RULE_NAMES:
        rules[rule] = {}
        for metric_index, metric in enumerate(METRIC_NAMES):
            values = _metric_array(rows, rule, metric)
            # Preserve the published family-rule bootstrap stream (rule index four).
            rules[rule][metric] = bootstrap_mean(
                values, BOOTSTRAP_B, seed + 404 + metric_index
            )
        rules[rule]["total_all_fail_case_events"] = float(
            sum(row["all_fail_case_events_by_rule"][rule] for row in rows)
        )
    return {"rules": rules}


def _metric_array(
    rows: list[dict[str, Any]], rule: str, metric: str
) -> np.ndarray:
    return np.asarray(
        [
            np.nan if row["rules"][rule][metric] is None else row["rules"][rule][metric]
            for row in rows
        ]
    )


def _selection_record(
    selections: dict[str, tuple[tuple[str, ...], ...]],
) -> dict[str, Any]:
    return {
        rule: {
            "pool": pools[0] if len(pools) == 1 else None,
        }
        for rule, pools in selections.items()
    }


def _run_split(
    task: str,
    task_index: int,
    split_seed: int,
    keys: tuple[str, ...],
    matrix: np.ndarray,
    case_ids: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, tuple[str, ...]]]:
    split = split_units(task, case_ids, split_seed)
    calibration_cases = matrix[:, split.calibration_columns]
    unit_ids = tuple(split.column_units[index] for index in split.calibration_columns)
    calibration_units = unit_matrix(calibration_cases, unit_ids)
    summary = calibration_summary(keys, calibration_units, FUNCTIONAL_FLOOR)
    p33 = empirical_p33(keys, calibration_cases, summary.best_model)
    evaluation = matrix[:, split.evaluation_columns]
    selections = _selections(summary)
    row = _evaluate_split_threshold(
        keys, evaluation, summary.best_model, p33, selections
    )
    row["threshold"] = p33
    record = {
        "split_seed": split_seed,
        "tau": p33,
        "anchor": summary.best_model,
        "n_calibration_units": len(split.calibration_units),
        "n_evaluation_units": len(split.evaluation_units),
        "functional_candidates": summary.functional_candidates,
        "quality_eligible": summary.quality_eligible,
        "selections": _selection_record(selections),
        "evaluation": row,
    }
    return {"empirical_p33": row}, record, {
        rule: pools[0] for rule, pools in selections.items()
    }


def _task_aggregates(
    threshold_rows: dict[str, list[dict[str, Any]]], task_index: int
) -> dict[str, Any]:
    rows = threshold_rows["empirical_p33"]
    return {
        "empirical_p33": {
            "threshold": {
                "kind": "calibration_empirical_p33",
                "mean": float(np.mean([row["threshold"] for row in rows])),
                "range": [
                    float(np.min([row["threshold"] for row in rows])),
                    float(np.max([row["threshold"] for row in rows])),
                ],
            },
            **_aggregate_threshold(rows, 100_000 * task_index),
        }
    }


def run_task(
    root: Path,
    task: str,
    task_index: int,
    split_count: int,
) -> dict[str, Any]:
    keys, matrix, case_ids, traces, extras = _load_task(root, task)
    threshold_rows: dict[str, list[dict[str, Any]]] = {
        "empirical_p33": []
    }
    primary_records: list[dict[str, Any]] = []
    selected_counts: dict[str, Counter[str]] = {
        rule: Counter() for rule in RULE_NAMES
    }
    for split_seed in range(split_count):
        rows, record, selections = _run_split(
            task, task_index, split_seed, keys, matrix, case_ids
        )
        for label, row in rows.items():
            threshold_rows[label].append(row)
        for rule, pool in selections.items():
            selected_counts[rule].update(pool)
        primary_records.append(record)
    return {
        "n_models": len(keys),
        "n_cases": int(matrix.shape[1]),
        "n_independent_units": len(set(split_units(task, case_ids, 0).column_units)),
        "source_files": source_files(traces, extras),
        "thresholds": _task_aggregates(threshold_rows, task_index),
        "primary_split_records": primary_records,
        "primary_selection_member_frequency": {
            rule: dict(sorted(counts.items()))
            for rule, counts in selected_counts.items()
        },
    }


def _protocol(split_count: int, tasks: tuple[str, ...] = TASKS) -> dict[str, Any]:
    return {
        "tasks": tasks,
        "excluded_tasks": ("msd_heart", "msd_spleen"),
        "pool_size": POOL_SIZE,
        "split_seeds": list(range(split_count)),
        "n_repeated_splits": split_count,
        "calibration_fraction": 0.5,
        "functional_floor": FUNCTIONAL_FLOOR,
        "primary_tau": "calibration_empirical_per_case_p33",
        "sensitivity_taus": (),
        "split_inference": (
            "Nonparametric bootstrap CI for the mean over split-level metrics; "
            "frequency intervals are computed separately by pool_low_meff."
        ),
        "bootstrap_B": BOOTSTRAP_B,
        "patient_level_split_tasks": ("acdc_lv", "brats_wt"),
    }


def run_analysis(
    root: Path,
    split_count: int = SPLIT_COUNT,
    scope: tuple[str, ...] = TASKS,
) -> dict[str, Any]:
    tasks = {}
    for task in scope:  # the fixed task index seeds each task, so restricted runs reproduce its rows
        tasks[task] = run_task(
            root, task, TASKS.index(task), split_count
        )
    return {
        "generated_at": now(),
        "status": "COMPLETE",
        "rule_definition_verbatim": RULE_DEFINITION,
        "primary_threshold_justification": THRESHOLD_JUSTIFICATION,
        "protocol": _protocol(split_count, scope),
        "tasks": tasks,
    }
