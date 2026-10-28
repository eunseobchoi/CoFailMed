"""Shared read-only trace loading and JSON helpers."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fmpool.case_ids import brats_patient_id

from analysis.trace_validation import Trace, apply_bundle_floor, load_trace

ARTIFACT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ARTIFACT_ROOT / "build/stats"
FUNCTIONAL_FLOOR = 0.30
BOOTSTRAP_B = 2000
RANDOM_SEED = 0
TASKS = ("kvasir", "acdc_lv", "brats_wt", "riga_cup", "riga_disc")


def collect_traces(root: Path, subdir: str, task: str) -> list[Trace]:
    task_root = root / subdir / task
    paths = sorted(task_root.glob("*/seed_*.json"))
    if not paths:
        raise FileNotFoundError(f"{task_root}: no per-case trace files")
    traces = [load_trace(path, root) for path in paths]
    wrong = [trace for trace in traces if trace.task != task]
    if wrong:
        raise ValueError(f"{wrong[0].source}: task mismatch for requested {task}")
    return traces


def primary_traces(root: Path, task: str) -> tuple[list[Trace], str]:
    expected_path = root / "within_cross" / f"{task}.json"
    with expected_path.open("r", encoding="utf-8") as handle:
        expected = json.load(handle)
    available = set(expected["fms_available_before_floor"])
    traces = [
        trace
        for trace in collect_traces(root, "per_case_dice", task)
        if trace.bundle in available
    ]
    kept = apply_bundle_floor(traces, FUNCTIONAL_FLOOR)
    return kept, str(expected_path.relative_to(root))


def aligned_matrix(
    traces: list[Trace],
    *,
    allow_partial: bool = False,
) -> tuple[list[tuple[str, int]], np.ndarray, tuple[str, ...]]:
    """Stack traces on their common test_ids; every trace must carry the same ID set
    unless ``allow_partial=True`` explicitly permits intersecting (dropping) cases."""
    common = set(traces[0].test_ids)
    for trace in traces[1:]:
        common.intersection_update(trace.test_ids)
    differing = [trace.source for trace in traces if set(trace.test_ids) != common]
    if differing and not allow_partial:
        raise ValueError(
            f"test_ids differ across traces ({len(common)} common; e.g. {differing[:3]}); "
            "pass allow_partial=True to intersect"
        )
    case_ids = tuple(sorted(common))
    if len(case_ids) < 2:
        raise ValueError("trace collection has fewer than two common test_ids")
    keys: list[tuple[str, int]] = []
    rows: list[np.ndarray] = []
    for trace in sorted(traces, key=lambda item: (item.bundle, item.seed)):
        lookup = dict(zip(trace.test_ids, trace.dice, strict=True))
        rows.append(np.asarray([lookup[case_id] for case_id in case_ids], dtype=float))
        keys.append((trace.bundle, trace.seed))
    return keys, np.stack(rows), case_ids


def subject_ids(task: str, case_ids: tuple[str, ...]) -> tuple[str, ...] | None:
    if task == "acdc_lv":
        return tuple(case_id.split("_frame", 1)[0] for case_id in case_ids)
    if task in {"brats_wt", "brats_wt_multimodal", "brats_multiclass", "brats_wt_3d"}:
        return tuple(brats_patient_id(case_id) for case_id in case_ids)
    return None


def aggregate_columns(
    matrix: np.ndarray,
    unit_ids: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    units = tuple(sorted(set(unit_ids)))
    unit_array = np.asarray(unit_ids)
    aggregated = np.stack(
        [np.mean(matrix[:, unit_array == unit], axis=1) for unit in units],
        axis=1,
    )
    return aggregated, units


def source_files(traces: list[Trace], extras: list[str] | None = None) -> list[str]:
    paths = [trace.source for trace in traces]
    if extras:
        paths.extend(extras)
    return sorted(set(paths))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
