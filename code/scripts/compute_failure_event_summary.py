#!/usr/bin/env python3
"""Five-task binary failure-event co-failure diagnostic.

The primary paper estimator uses Pearson correlation of per-case Dice traces.
This CPU-only diagnostic asks the complementary event question on the same
released traces: after the primary functional floor, do binary failure events
co-occur more strongly for same-bundle/release pairs than for cross-bundle
pairs?

For each task, failure thresholds are empirical percentiles over all
model-case Dice values in the functional pool. For each member pair we compute
the binary-event Pearson/phi correlation and the joint-failure rate, then
average those quantities under the primary same/cross grouping rule.
Thresholds are re-estimated in each paired resample, conditional on the retained
pool. ACDC/BraTS primary intervals resample whole subjects; case intervals remain
labelled as diagnostics. This does not account for prior pool/recipe selection.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fmpool.estimators import _bootstrap_case_indices, functional_floor_filter  # noqa: E402

import compute_all as ca  # noqa: E402


SCHEMA_VERSION = "fmpool_failure_event_summary_v2"
PERCENTILES = (25, 33, 50)
DEFAULT_BOOTSTRAP_B = 1000
DEFAULT_BOOTSTRAP_SEED = 0


def _now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_json_value(payload), fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")


def _is_same_group(fm_a: str, fm_b: str) -> bool:
    return fm_a == fm_b or frozenset({fm_a, fm_b}) in ca.DINOV2_FAMILY_PAIRS


def _mean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else float("nan")


def _ci95(values: list[float]) -> list[float | None]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return [None, None]
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return [float(lo), float(hi)]


def _empirical_threshold(per_case: dict[tuple[str, int], np.ndarray], percentile: float,
                         indices: np.ndarray | None = None) -> float:
    data = np.stack([np.asarray(per_case[k], dtype=float) for k in sorted(per_case)])
    if data.ndim != 2 or not np.isfinite(data).all() or not 0 <= percentile <= 100:
        raise ValueError("expected finite aligned traces and percentile in [0, 100]")
    if indices is not None:
        data = data[:, indices]
    return float(np.percentile(data, percentile))


def _event_stats(
    per_case: dict[tuple[str, int], np.ndarray],
    *,
    threshold: float,
    indices: np.ndarray | None = None,
) -> dict[str, Any]:
    keys = sorted(per_case)
    scores = np.stack([np.asarray(per_case[k], dtype=np.float64) for k in keys])
    if scores.ndim != 2 or not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("expected finite aligned traces and threshold")
    if indices is not None:
        scores = scores[:, indices]
    if scores.shape[1] == 0:
        raise ValueError("no sampled cases")
    fail = (scores < threshold).astype(np.float64)
    n_members, n_cases = fail.shape
    pair_i, pair_j = np.triu_indices(n_members, k=1)
    same_mask = np.asarray(
        [_is_same_group(keys[i][0], keys[j][0]) for i, j in zip(pair_i, pair_j, strict=True)],
        dtype=bool,
    )

    centered = fail - fail.mean(axis=1, keepdims=True)
    ss = np.sum(centered * centered, axis=1)
    denom = np.sqrt(np.outer(ss, ss))
    with np.errstate(invalid="ignore", divide="ignore"):
        phi_matrix = (centered @ centered.T) / denom
    phi_matrix[denom <= 1e-12] = np.nan
    joint_matrix = (fail @ fail.T) / float(n_cases)

    pair_phi = phi_matrix[pair_i, pair_j]
    pair_joint = joint_matrix[pair_i, pair_j]
    same_phi = pair_phi[same_mask].tolist()
    cross_phi = pair_phi[~same_mask].tolist()
    same_joint = pair_joint[same_mask].tolist()
    cross_joint = pair_joint[~same_mask].tolist()

    same_phi_mean = _mean(same_phi)
    cross_phi_mean = _mean(cross_phi)
    same_joint_mean = _mean(same_joint)
    cross_joint_mean = _mean(cross_joint)
    return {
        "same_fail_phi": same_phi_mean,
        "cross_fail_phi": cross_phi_mean,
        "gap_fail_phi": (
            same_phi_mean - cross_phi_mean
            if math.isfinite(same_phi_mean) and math.isfinite(cross_phi_mean)
            else float("nan")
        ),
        "same_joint_fail_rate": same_joint_mean,
        "cross_joint_fail_rate": cross_joint_mean,
        "joint_fail_rate_ratio": (
            same_joint_mean / cross_joint_mean
            if math.isfinite(same_joint_mean) and cross_joint_mean > 0
            else float("nan")
        ),
        "n_values_equal_threshold": int(np.sum(scores == threshold)),
        "failure_rule": "score < threshold; ties are passes",
        "n_constant_failure_members": int(np.sum(ss <= 1e-12)),
        "n_same_pairs": int(np.sum(same_mask)),
        "n_cross_pairs": int(np.sum(~same_mask)),
        "n_same_valid_phi_pairs": int(sum(math.isfinite(v) for v in same_phi)),
        "n_cross_valid_phi_pairs": int(sum(math.isfinite(v) for v in cross_phi)),
    }


def _bootstrap_events(per_case: dict[tuple[str, int], np.ndarray], *, percentile: float,
                      n_boot: int, seed: int,
                      case_to_subject: tuple[str, ...] | None = None) -> dict[str, Any]:
    gaps, ratios, thresholds, pair_counts = [], [], [], []
    n_cases = len(next(iter(per_case.values())))
    for indices in _bootstrap_case_indices(n_cases, n_boot, seed, case_to_subject):
        threshold = _empirical_threshold(per_case, percentile, indices)
        boot = _event_stats(per_case, threshold=threshold, indices=indices)
        thresholds.append(threshold)
        pair_counts.append({"same": boot["n_same_valid_phi_pairs"],
                            "cross": boot["n_cross_valid_phi_pairs"]})
        if math.isfinite(boot["gap_fail_phi"]):
            gaps.append(boot["gap_fail_phi"])
        if math.isfinite(boot["joint_fail_rate_ratio"]):
            ratios.append(boot["joint_fail_rate_ratio"])
    return {
        "B_requested": n_boot, "B_completed_gap": len(gaps), "B_completed_ratio": len(ratios),
        "seed": seed, "unit": "subject" if case_to_subject is not None else "case",
        "conditioning": "observed retained pool; empirical percentile re-estimated per draw",
        "gap_fail_phi_95ci": _ci95(gaps), "joint_fail_rate_ratio_95ci": _ci95(ratios),
        "thresholds": thresholds, "valid_phi_pair_counts": pair_counts,
        "pair_policy": "phi means use defined pairs per draw; counts expose changing eligibility",
    }


def _summarize_task(
    results_root: Path,
    task: str,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    fm_jsons, _nnunet = ca._collect_task_jsons(results_root, task)
    ca._validate_alignment(fm_jsons)
    per_case_all = ca._as_per_case_map(fm_jsons)
    per_case = functional_floor_filter(per_case_all, ca.FUNCTIONAL_FLOOR)
    if len(per_case) < 2:
        raise RuntimeError(f"{task}: fewer than two traces after functional floor")

    meta = ca._case_metadata(task, fm_jsons[0].test_ids)
    n_cases = len(next(iter(per_case.values())))
    unit_ids = tuple(str(x) for x in meta["unit_ids"])
    rows: list[dict[str, Any]] = []
    for percentile in PERCENTILES:
        threshold = _empirical_threshold(per_case, percentile)
        point = _event_stats(per_case, threshold=threshold)
        case_boot = _bootstrap_events(per_case, percentile=percentile, n_boot=n_boot, seed=seed)
        primary_boot = (_bootstrap_events(per_case, percentile=percentile, n_boot=n_boot, seed=seed,
                                          case_to_subject=unit_ids)
                        if meta["cluster_unit"] is not None else case_boot)
        rows.append({
            "percentile": percentile, "threshold": threshold, **point,
            "case_bootstrap_95ci_gap_fail_phi": case_boot["gap_fail_phi_95ci"],
            "case_bootstrap_95ci_joint_fail_rate_ratio": case_boot["joint_fail_rate_ratio_95ci"],
            "case_bootstrap": case_boot,
            "primary_bootstrap_95ci_gap_fail_phi": primary_boot["gap_fail_phi_95ci"],
            "primary_bootstrap_95ci_joint_fail_rate_ratio": primary_boot["joint_fail_rate_ratio_95ci"],
            "bootstrap": primary_boot,
        })

    source_files = ca._rel_paths([j.path for j in fm_jsons], results_root)
    return {
        "task": task,
        "functional_floor": float(ca.FUNCTIONAL_FLOOR),
        "interpretation": "exploratory finite-panel diagnostic conditional on floor/recipe selection",
        "primary_resampling_unit": meta["cluster_unit"] or meta["case_unit"],
        "n_units": int(meta["n_units"]),
        "threshold_definition": (
            "empirical percentile over all model-case Dice values in the "
            "functional pool retained after Dice-floor filtering"
        ),
        "fms_after_floor": sorted({fm for fm, _seed in per_case}),
        "n_members_after_floor": int(len(per_case)),
        "n_cases": int(n_cases),
        "rows": rows,
        "source_files": source_files,
    }


def _supported_range(task_rows: dict, field: str) -> dict:
    values = {task: record["rows"][1][field] for task, record in task_rows.items()}
    finite = {task: value for task, value in values.items()
              if value is not None and math.isfinite(value)}
    undefined = sorted(set(values) - set(finite))
    bounds = [min(finite.values()), max(finite.values())] if finite else [None, None]
    return {"range": bounds if not undefined else [None, None],
            "defined_range": bounds, "n_defined": len(finite),
            "undefined_tasks": undefined}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True,
                        help="merged results root (contains per_case_dice/)")
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_BOOTSTRAP_B)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()

    task_rows = {
        task: _summarize_task(
            args.results_root,
            task,
            n_boot=args.n_boot,
            seed=args.seed + task_index * 1009,
        )
        for task_index, task in enumerate(ca.TASKS)
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rig_version": ca.STATISTICS_RIG_VERSION,
        "generated_at": _now(),
        "results_root": str(args.results_root),
        "same_group_rule": "same FM plus DINOv2-B/S, matching the primary grouping",
        "description": (
            "Binary failure-event same/cross diagnostic computed from released "
            "per-case Dice traces after the primary functional floor."
        ),
        "percentiles": list(PERCENTILES),
        "bootstrap_B": int(args.n_boot),
        "bootstrap_seed": int(args.seed),
        "task_rows": task_rows,
        "summary": {
            "all_p33_primary_gap_cis_exclude_zero": all(
                ci[0] is not None and ci[1] is not None
                and math.isfinite(ci[0]) and math.isfinite(ci[1])
                and (ci[0] > 0 or ci[1] < 0)
                for ci in (task_rows[task]["rows"][1]["primary_bootstrap_95ci_gap_fail_phi"]
                           for task in ca.TASKS)
            ),
            "all_p33_primary_gap_cis_positive": bool(
                all(
                    (row := task_rows[task]["rows"][1])[
                        "primary_bootstrap_95ci_gap_fail_phi"
                    ][0]
                    is not None
                    and row["primary_bootstrap_95ci_gap_fail_phi"][0] > 0
                    for task in ca.TASKS
                )
            ),
            **{"p33_gap_" + key: value for key, value in
               _supported_range(task_rows, "gap_fail_phi").items()},
            **{"p33_joint_fail_ratio_" + key: value for key, value in
               _supported_range(task_rows, "joint_fail_rate_ratio").items()},
        },
    }
    _write_json(args.out, payload)
    for task in ca.TASKS:
        row = task_rows[task]["rows"][1]
        ci = row["primary_bootstrap_95ci_gap_fail_phi"]
        print(
            f"{task}: P33 phi gap={row['gap_fail_phi']:.3f} "
            f"ci={ci} "
            f"joint ratio={row['joint_fail_rate_ratio']:.2f}"
        )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
