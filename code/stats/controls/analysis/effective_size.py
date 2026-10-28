"""Effective pool size for the control pools: trace loading, support checks, pool summaries, bootstrap."""
from __future__ import annotations
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from analysis.common import subject_ids

FIXED_THRESHOLDS = (0.70, 0.80, 0.85, 0.90, 0.95)
MIN_FAILURES = 5
MIN_PASSES = 5
UNDEFINED_REASON = "undefined — zero/near-zero indicator variance"


@dataclass(frozen=True)
class TraceCollection:
    config: str
    task: str
    member_ids: tuple[str, ...]
    bundle_ids: tuple[str, ...]
    matrix: np.ndarray
    case_ids: tuple[str, ...]
    cluster_ids: tuple[str, ...] | None
    provenance: tuple[str, ...]
    evidence_scope: str
    note: str | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
            raise ValueError(f"{self.config}/{self.task}: invalid matrix shape {matrix.shape}")
        if matrix.shape[0] != len(self.member_ids):
            raise ValueError(f"{self.config}/{self.task}: member count mismatch")
        if matrix.shape[0] != len(self.bundle_ids):
            raise ValueError(f"{self.config}/{self.task}: bundle count mismatch")
        if matrix.shape[1] != len(self.case_ids):
            raise ValueError(f"{self.config}/{self.task}: case count mismatch")
        if self.cluster_ids is not None and len(self.cluster_ids) != len(self.case_ids):
            raise ValueError(f"{self.config}/{self.task}: cluster count mismatch")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{self.config}/{self.task}: non-finite Dice")


@dataclass(frozen=True)
class PoolSpec:
    name: str
    nominal_m: int
    pool_members: np.ndarray
    pair_left: np.ndarray
    pair_right: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _trace_values(payload: dict[str, Any], path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    raw = payload.get("dice", payload.get("per_case_dice"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: missing Dice list")
    values = np.asarray(raw, dtype=float)
    ids = payload.get("test_ids")
    if ids is None:
        ids = [f"position_{index:05d}" for index in range(len(values))]
    if not isinstance(ids, list) or len(ids) != len(values):
        raise ValueError(f"{path}: test_ids/Dice mismatch")
    return values, tuple(str(item) for item in ids)


def _load_trace_paths(
    paths: Sequence[Path],
    *,
    config: str,
    task: str,
    evidence_scope: str,
    cluster_task: str | None = None,
    note: str | None = None,
) -> TraceCollection:
    rows: list[tuple[str, str, tuple[str, ...], np.ndarray, str]] = []
    for path in sorted(paths):
        payload = _read_json(path)
        payload_task = str(payload.get("task", task))
        if payload_task != task:
            raise ValueError(f"{path}: task {payload_task!r} != {task!r}")
        fm = str(payload.get("fm", path.parent.name))
        seed = int(payload.get("seed", path.stem.split("_")[-1]))
        values, ids = _trace_values(payload, path)
        rows.append((f"{fm}/seed_{seed}", fm, ids, values, str(path)))
    if not rows:
        raise FileNotFoundError(f"{config}/{task}: no trace paths")
    canonical_ids = rows[0][2]
    matrix_rows = []
    for member, _, ids, values, source in rows:
        if len(set(ids)) != len(ids):
            raise ValueError(f"{source}: duplicate test IDs")
        if set(ids) != set(canonical_ids):
            raise ValueError(f"{source}: case set mismatch for {member}")
        lookup = dict(zip(ids, values, strict=True))
        matrix_rows.append(np.asarray([lookup[item] for item in canonical_ids]))
    member_ids = tuple(row[0] for row in rows)
    if len(set(member_ids)) != len(member_ids):
        raise ValueError(f"{config}/{task}: duplicate operational members")
    clusters = (
        subject_ids(cluster_task, canonical_ids) if cluster_task is not None else None
    )
    return TraceCollection(
        config=config,
        task=task,
        member_ids=member_ids,
        bundle_ids=tuple(row[1] for row in rows),
        matrix=np.stack(matrix_rows),
        case_ids=canonical_ids,
        cluster_ids=clusters,
        provenance=tuple(row[4] for row in rows),
        evidence_scope=evidence_scope,
        note=note,
    )


def threshold_union(matrix: np.ndarray) -> list[dict[str, Any]]:
    pooled = np.asarray(matrix, dtype=float).ravel()
    percentile_items = (
        ("P25", float(np.percentile(pooled, 25.0))),
        ("P33", float(np.percentile(pooled, 33.0))),
        ("P50", float(np.percentile(pooled, 50.0))),
    )
    items = list(percentile_items) + [(f"fixed_{value:.2f}", value) for value in FIXED_THRESHOLDS]
    merged: list[dict[str, Any]] = []
    for label, value in sorted(items, key=lambda item: (item[1], item[0])):
        match = next((row for row in merged if row["value"] == value), None)
        if match is None:
            merged.append({"value": value, "sources": [label]})
        else:
            match["sources"].append(label)
    return merged


def member_support(matrix: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    failures = np.sum(matrix < threshold, axis=1)
    passes = matrix.shape[1] - failures
    valid = (failures >= MIN_FAILURES) & (passes >= MIN_PASSES)
    return valid, failures, passes


def _pool_spec_diverse4(bundle_ids: Sequence[str]) -> PoolSpec:
    by_bundle: dict[str, list[int]] = {}
    for index, bundle in enumerate(bundle_ids):
        by_bundle.setdefault(bundle, []).append(index)
    pools: list[tuple[int, ...]] = []
    for bundles in itertools.combinations(sorted(by_bundle), 4):
        pools.extend(itertools.product(*(by_bundle[bundle] for bundle in bundles)))
    members = np.asarray(pools, dtype=np.int32)
    local_pairs = np.asarray(list(itertools.combinations(range(4), 2)), dtype=np.int32)
    return PoolSpec(
        name="diverse4_enumeration",
        nominal_m=4,
        pool_members=members,
        pair_left=members[:, local_pairs[:, 0]],
        pair_right=members[:, local_pairs[:, 1]],
    )


def _correlation_matrix(failures: np.ndarray) -> np.ndarray:
    data = np.asarray(failures, dtype=float)
    centered = data - np.mean(data, axis=1, keepdims=True)
    norm = np.sqrt(np.sum(centered * centered, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (centered @ centered.T) / np.outer(norm, norm)


def pool_summary(
    matrix: np.ndarray, threshold: float, spec: PoolSpec
) -> dict[str, Any]:
    valid_members, failures_per_member, passes_per_member = member_support(matrix, threshold)
    valid_pools = np.all(valid_members[spec.pool_members], axis=1)
    n_valid = int(np.sum(valid_pools))
    base = {
        "n_pools_total": int(len(spec.pool_members)),
        "n_pools_supported": n_valid,
        "n_pools_finite": 0,
        "n_pools_numerically_ineligible": 0,
        "n_pools_undefined": int(len(spec.pool_members) - n_valid),
        "member_failures": failures_per_member.astype(int).tolist(),
        "member_passes": passes_per_member.astype(int).tolist(),
    }
    if n_valid == 0:
        return {
            **base,
            "status": UNDEFINED_REASON,
            "rho_fail": None,
            "M_eff": None,
        }
    correlation = _correlation_matrix(matrix < threshold)
    rhos = np.mean(
        correlation[spec.pair_left[valid_pools], spec.pair_right[valid_pools]],
        axis=1,
    )
    denominator = 1.0 + (spec.nominal_m - 1) * rhos
    # Match fmpool.estimators: support alone does not ensure a defined size.
    eligible = np.isfinite(rhos) & (denominator > 1e-12)
    meffs = np.divide(
        spec.nominal_m, denominator,
        out=np.full(rhos.shape, np.nan), where=eligible,
    )
    finite = eligible & np.isfinite(meffs)
    n_finite = int(np.sum(finite))
    base.update({
        "n_pools_finite": n_finite,
        "n_pools_numerically_ineligible": n_valid - n_finite,
        "n_pools_undefined": int(len(spec.pool_members) - n_finite),
    })
    if not np.any(finite):
        return {
            **base,
            "status": "undefined — nonpositive/near-zero effective-size denominator",
            "rho_fail": None,
            "M_eff": None,
        }
    return {
        **base,
        "status": "support-valid",
        "rho_fail": float(np.mean(rhos[finite])),
        "M_eff": float(np.mean(meffs[finite])),
    }


def _resample_indices(
    rng: np.random.Generator,
    n_cases: int,
    cluster_ids: tuple[str, ...] | None,
) -> np.ndarray:
    if cluster_ids is None:
        return rng.integers(0, n_cases, size=n_cases)
    clusters = tuple(sorted(set(cluster_ids)))
    picked = rng.integers(0, len(clusters), size=len(clusters))
    cluster_array = np.asarray(cluster_ids)
    pieces = [np.flatnonzero(cluster_array == clusters[index]) for index in picked]
    return np.concatenate(pieces)


def bootstrap_pool_rows(
    matrix: np.ndarray,
    thresholds: Sequence[float],
    spec: PoolSpec,
    *,
    b: int,
    seed: int,
    cluster_ids: tuple[str, ...] | None,
) -> dict[float, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rho_draws = {threshold: np.full(b, np.nan) for threshold in thresholds}
    meff_draws = {threshold: np.full(b, np.nan) for threshold in thresholds}
    for draw in range(b):
        sampled = matrix[:, _resample_indices(rng, matrix.shape[1], cluster_ids)]
        for threshold in thresholds:
            summary = pool_summary(sampled, threshold, spec)
            if summary["status"] == "support-valid":
                rho_draws[threshold][draw] = summary["rho_fail"]
                meff_draws[threshold][draw] = summary["M_eff"]
    output: dict[float, dict[str, Any]] = {}
    for threshold in thresholds:
        rho = rho_draws[threshold][np.isfinite(rho_draws[threshold])]
        meff = meff_draws[threshold][np.isfinite(meff_draws[threshold])]
        output[threshold] = {
            "B_requested": b,
            "B_finite": int(min(len(rho), len(meff))),
            "rho_fail_ci95": (
                [float(value) for value in np.percentile(rho, [2.5, 97.5])]
                if len(rho)
                else None
            ),
            "M_eff_ci95": (
                [float(value) for value in np.percentile(meff, [2.5, 97.5])]
                if len(meff)
                else None
            ),
        }
    return output


def _threshold_record(
    candidate: dict[str, Any],
    point: dict[str, Any],
    bootstrap: dict[str, Any] | None,
) -> dict[str, Any]:
    if point["status"] != "support-valid":
        return {
            "threshold": candidate,
            "status": point["status"],
            "rho_fail": {"value": None, "ci95": None, "evidence": "MEASURED"},
            "M_eff": {"value": None, "ci95": None, "evidence": "DERIVED"},
            "support": point,
        }
    return {
        "threshold": candidate,
        "status": "support-valid",
        "rho_fail": {
            "value": point["rho_fail"],
            "ci95": None if bootstrap is None else bootstrap["rho_fail_ci95"],
            "evidence": "MEASURED",
        },
        "M_eff": {
            "value": point["M_eff"],
            "ci95": None if bootstrap is None else bootstrap["M_eff_ci95"],
            "evidence": "DERIVED",
        },
        "support": point,
        "bootstrap": bootstrap,
    }
