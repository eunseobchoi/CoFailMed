"""M_eff scoring of recorded architecture-diversity pools (used by ``analysis.pool_low_meff``)."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from analysis.common import source_files
from analysis.effective_size import _pool_spec_fixed, pool_summary
from analysis.pool_config import POOL_SIZE
from analysis.pool_core import (
    architecture_family,
    architecture_pool,
    calibration_summary,
    evaluation_vectors,
    split_units,
    unit_matrix,
)
from analysis.pool_evaluation import point
from analysis.pool_repeated import empirical_p33
from analysis.pool_task import _load_task

BOOTSTRAP_B = 2000
RANDOM_SEED = 0
LOW_MEFF_CUTOFF = 2.0
FLOAT_TOLERANCE = 1e-12


def sha256_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def assert_float_equal(
    calculated: float | None,
    recorded: float | None,
    label: str,
) -> None:
    if calculated is None or recorded is None:
        if calculated is not recorded:
            raise ValueError(f"{label}: calculated={calculated}, recorded={recorded}")
        return
    if not np.isfinite(calculated) or not np.isfinite(recorded):
        raise ValueError(f"{label}: nonfinite comparison")
    if abs(float(calculated) - float(recorded)) > FLOAT_TOLERANCE:
        raise ValueError(f"{label}: calculated={calculated}, recorded={recorded}")


def check_recorded_selection(
    *,
    task: str,
    record: dict[str, Any],
    keys: tuple[str, ...],
    matrix: np.ndarray,
    case_ids: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    split_seed = int(record["split_seed"])
    split = split_units(task, case_ids, split_seed)
    calibration_cases = matrix[:, split.calibration_columns]
    calibration_unit_ids = tuple(
        split.column_units[index] for index in split.calibration_columns
    )
    calibration_units = unit_matrix(calibration_cases, calibration_unit_ids)
    summary = calibration_summary(keys, calibration_units, 0.30)
    threshold = empirical_p33(keys, calibration_cases, summary.best_model)
    recorded_pool = tuple(
        record["selections"]["architecture_diversity"]["pool"]
    )
    reproduced_pool = architecture_pool(
        summary.functional_candidates,
        summary.means,
        POOL_SIZE,
    )
    label = f"{task}/split {split_seed}"
    assert_float_equal(threshold, record["tau"], f"{label} P33")
    if summary.best_model != record["anchor"]:
        raise ValueError(f"{label}: anchor reproduction failed")
    if list(summary.functional_candidates) != record["functional_candidates"]:
        raise ValueError(f"{label}: functional candidate reproduction failed")
    if list(summary.quality_eligible) != record["quality_eligible"]:
        raise ValueError(f"{label}: quality-gate reproduction failed")
    if reproduced_pool != recorded_pool:
        raise ValueError(
            f"{label}: architecture selection reproduction failed: "
            f"{reproduced_pool} != {recorded_pool}"
        )
    if len({architecture_family(item) for item in recorded_pool}) != POOL_SIZE:
        raise ValueError(f"{label}: recorded pool is not architecture-diverse")
    if (
        len(split.calibration_units) != record["n_calibration_units"]
        or len(split.evaluation_units) != record["n_evaluation_units"]
    ):
        raise ValueError(f"{label}: independent-unit split reproduction failed")

    evaluation = matrix[:, split.evaluation_columns]
    vectors = evaluation_vectors(
        keys,
        evaluation,
        (recorded_pool,),
        record["anchor"],
        threshold,
    )
    reproduced_metrics = point(vectors)
    recorded_evaluation = record["evaluation"]
    for metric, value in reproduced_metrics.items():
        assert_float_equal(
            value,
            recorded_evaluation["rules"]["architecture_diversity"][metric],
            f"{label} evaluation {metric}",
        )
    if evaluation.shape[1] != recorded_evaluation["n_evaluation_cases"]:
        raise ValueError(f"{label}: evaluation case count reproduction failed")
    if int(np.sum(vectors.anchor_fail)) != recorded_evaluation[
        "n_anchor_failure_cases"
    ]:
        raise ValueError(f"{label}: anchor failure reproduction failed")
    assert_float_equal(
        float(np.sum(vectors.all_fail)),
        recorded_evaluation["all_fail_case_events_by_rule"][
            "architecture_diversity"
        ],
        f"{label} all-fail events",
    )
    return evaluation, recorded_pool, {
        "calibration_units_sha256": sha256_strings(split.calibration_units),
        "evaluation_units_sha256": sha256_strings(split.evaluation_units),
        "n_calibration_units": len(split.calibration_units),
        "n_evaluation_units": len(split.evaluation_units),
        "n_evaluation_cases": int(evaluation.shape[1]),
        "p33": threshold,
        "anchor": record["anchor"],
    }


def split_meff_record(
    evaluation: np.ndarray,
    keys: tuple[str, ...],
    pool: tuple[str, ...],
    threshold: float,
    split_metadata: dict[str, Any],
    split_seed: int,
) -> dict[str, Any]:
    index = {key: position for position, key in enumerate(keys)}
    selected = evaluation[[index[key] for key in pool]]
    scored = pool_summary(
        selected,
        threshold,
        _pool_spec_fixed(POOL_SIZE, "recorded_architecture_diversity"),
    )
    # support_valid includes numerical eligibility; support_adequate tracks counts alone.
    support_valid = scored["status"] == "support-valid"
    meff = scored["M_eff"] if support_valid else None
    low = bool(support_valid and float(meff) < LOW_MEFF_CUTOFF)
    outcome = (
        "detected_low_M_eff"
        if low
        else "supported_not_low_M_eff"
        if support_valid
        else "unsupported"
    )
    return {
        "split_seed": split_seed,
        "selection_evidence_tag": "CITED",
        "measurement_evidence_tag": "MEASURED",
        "pool": list(pool),
        "M_nominal": POOL_SIZE,
        "low_M_eff_cutoff": LOW_MEFF_CUTOFF,
        "support_status": scored["status"],
        "support_valid": support_valid,
        "support_adequate": scored["n_pools_supported"] > 0,
        "member_failures": scored["member_failures"],
        "member_passes": scored["member_passes"],
        "rho_fail": scored["rho_fail"],
        "M_eff": meff,
        "low_M_eff": low,
        "outcome": outcome,
        **split_metadata,
    }


def interval(values: np.ndarray) -> list[float]:
    return [
        float(value) for value in np.percentile(values, [2.5, 97.5])
    ]


def summarize_records(
    records: list[dict[str, Any]],
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    valid = np.asarray([record["support_valid"] for record in records], dtype=bool)
    low = np.asarray([record["low_M_eff"] for record in records], dtype=bool)
    n_total = len(records)
    n_valid = int(np.sum(valid))
    n_low = int(np.sum(low))
    n_not_low = int(np.sum(valid & ~low))
    n_unsupported = n_total - n_valid
    sampled_valid = valid[bootstrap_indices]
    sampled_low = low[bootstrap_indices]
    support_samples = np.mean(sampled_valid, axis=1)
    all_samples = np.mean(sampled_low, axis=1)
    valid_denominators = np.sum(sampled_valid, axis=1)
    conditional_samples = np.divide(
        np.sum(sampled_low, axis=1),
        valid_denominators,
        out=np.full(valid_denominators.shape, np.nan, dtype=float),
        where=valid_denominators > 0,
    )
    conditional_finite = conditional_samples[np.isfinite(conditional_samples)]
    if n_valid == 0 or conditional_finite.size == 0:
        conditional = {"value": None, "ci95": None, "denominator": 0}
        distribution = None
    else:
        meffs = np.asarray(
            [record["M_eff"] for record in records if record["support_valid"]],
            dtype=float,
        )
        conditional = {
            "value": n_low / n_valid,
            "ci95": interval(conditional_finite),
            "denominator": n_valid,
        }
        distribution = {
            "n": n_valid,
            "min": float(np.min(meffs)),
            "q25": float(np.percentile(meffs, 25.0)),
            "median": float(np.median(meffs)),
            "q75": float(np.percentile(meffs, 75.0)),
            "max": float(np.max(meffs)),
        }
    return {
        "n_splits": n_total,
        "n_numerically_ineligible": sum(
            record.get("support_adequate", record["support_valid"])
            and not record["support_valid"] for record in records
        ),
        "outcome_counts": {
            "detected_low_M_eff": n_low,
            "supported_not_low_M_eff": n_not_low,
            "unsupported": n_unsupported,
        },
        "primary_low_M_eff_fraction_all_splits": {
            "value": n_low / n_total,
            "ci95": interval(all_samples),
            "numerator": n_low,
            "denominator": n_total,
            "unsupported_counted_as_low_event": False,
            "evidence_tag": "MEASURED",
        },
        "low_M_eff_fraction_support_valid": {
            **conditional,
            "numerator": n_low,
            "evidence_tag": "MEASURED",
        },
        "support_valid_fraction": {
            "value": n_valid / n_total,
            "ci95": interval(support_samples),
            "numerator": n_valid,
            "denominator": n_total,
            "evidence_tag": "MEASURED",
        },
        "unsupported_fraction": {
            "value": n_unsupported / n_total,
            "ci95": [
                1.0 - interval(support_samples)[1],
                1.0 - interval(support_samples)[0],
            ],
            "numerator": n_unsupported,
            "denominator": n_total,
            "evidence_tag": "MEASURED",
        },
        "supported_not_low_M_eff_fraction_all_splits": {
            "value": n_not_low / n_total,
            "numerator": n_not_low,
            "denominator": n_total,
            "evidence_tag": "MEASURED",
        },
        "support_valid_M_eff_distribution": distribution,
    }


def analyze_task(
    *,
    task: str,
    task_payload: dict[str, Any],
    results_root: Path,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    keys, matrix, case_ids, traces, extras = _load_task(results_root, task)
    calculated_sources = source_files(traces, extras)
    if calculated_sources != task_payload["source_files"]:
        raise ValueError(f"{task}: source provenance does not exactly reproduce")
    records: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for record in task_payload["primary_split_records"]:
        evaluation, pool, metadata = check_recorded_selection(
            task=task,
            record=record,
            keys=keys,
            matrix=matrix,
            case_ids=case_ids,
        )
        selected_counts.update(pool)
        records.append(
            split_meff_record(
                evaluation,
                keys,
                pool,
                float(record["tau"]),
                metadata,
                int(record["split_seed"]),
            )
        )
    recorded_frequency = task_payload["primary_selection_member_frequency"][
        "architecture_diversity"
    ]
    if dict(sorted(selected_counts.items())) != recorded_frequency:
        raise ValueError(f"{task}: recorded member-frequency reproduction failed")
    return {
        "status": "PASS",
        "selection_reproduction": {
            "status": "PASS",
            "n_split_records_reproduced": len(records),
            "split_seeds": [record["split_seed"] for record in records],
            "recorded_member_frequency_reproduced": True,
            "source_files_reproduced": True,
            "evidence_tag": "CITED",
        },
        "summary": summarize_records(records, bootstrap_indices),
        "split_records": records,
    }
