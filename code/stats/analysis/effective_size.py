"""Primary frozen-encoder configuration-wise effective pool size.

Validates aligned traces, enumerates pools and reports threshold-specific
effective sizes with support checks and bootstrap intervals.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from analysis.common import subject_ids

DEFAULT_OUTPUT_DIR = Path("build/stats/effective_size")


def set_roots(traces_root: Path, stats_root: Path | None = None) -> None:
    """Set trace and statistics roots for the primary effective-size analysis."""
    # Module-level roots: one traces/stats root pair per process.
    global RELEASED_ROOT, STATS_ROOT
    RELEASED_ROOT = Path(traces_root)
    STATS_ROOT = Path(stats_root) if stats_root is not None else RELEASED_ROOT / "stats"


def load_released_task(task: str) -> TraceCollection:
    if task not in PRIMARY_TASKS:
        raise ValueError(f"unknown primary task {task}")
    meff_path = STATS_ROOT / "m_eff" / f"{task}.json"
    meff = _read_json(meff_path)
    bundles = tuple(item["fm"] for item in meff["mono_pools"])
    collection = _load_trace_paths(
        _seed_paths(RELEASED_ROOT / "per_case_dice", task, bundles),
        config="released_1x1",
        task=task,
        evidence_scope="released primary",
        cluster_task=task if task in {"acdc_lv", "brats_wt"} else None,
    )
    return TraceCollection(
        **{**collection.__dict__, "provenance": collection.provenance + (str(meff_path),)}
    )


def build_metrics(*, b: int, seed: int) -> dict[str, Any]:
    rows = [
        analyze_collection(load_released_task(task), b=b, seed=seed + index * 100_000,
                           include_diverse4=True, include_full=True)
        for index, task in enumerate(PRIMARY_TASKS)
    ]
    return {
        "schema_version": "e1_meff_configs_v1",
        "generated_at": utc_now(),
        "status": "complete",
        "scope": "five primary frozen-encoder 1x1 configurations",
        "input_roots": {"traces_root": str(RELEASED_ROOT), "stats_root": str(STATS_ROOT)},
        "protocol": {
            "path": str(Path(__file__).resolve()),
            "threshold_union": "pooled P25/P33/P50 plus {0.70,0.80,0.85,0.90,0.95}",
            "primary_threshold": "P33",
            "sensitivity_threshold": "highest support-valid candidate",
            "per_member_min_failures": MIN_FAILURES,
            "per_member_min_passes": MIN_PASSES,
            "bootstrap_B": b,
            "bootstrap_seed": seed,
            "undefined_text": UNDEFINED_REASON,
            "no_support_text": NO_SUPPORT_REASON,
        },
        "configurations": rows,
    }

FIXED_THRESHOLDS = (0.70, 0.80, 0.85, 0.90, 0.95)
PRIMARY_TASKS = ("kvasir", "acdc_lv", "brats_wt", "riga_cup", "riga_disc")
MIN_FAILURES = 5
MIN_PASSES = 5
UNDEFINED_REASON = "undefined — zero/near-zero indicator variance"
NO_SUPPORT_REASON = "no threshold with adequate support"


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


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


def _seed_paths(root: Path, task: str, bundles: Iterable[str] | None = None) -> list[Path]:
    allowed = None if bundles is None else set(bundles)
    paths = sorted((root / task).glob("*/seed_*.json"))
    if allowed is not None:
        paths = [path for path in paths if path.parent.name in allowed]
    return paths


def _collection_from_embedded(
    *,
    config: str,
    task: str,
    member_map: dict[str, list[float]],
    case_ids: Sequence[str] | None,
    provenance: Path,
    evidence_scope: str,
    note: str | None = None,
) -> TraceCollection:
    members = tuple(sorted(member_map))
    rows = np.stack([np.asarray(member_map[item], dtype=float) for item in members])
    if case_ids is None:
        ids = tuple(f"aggregate_position_{index:05d}" for index in range(rows.shape[1]))
    else:
        ids = tuple(str(item) for item in case_ids)
    bundles = tuple(item.rsplit("_seed", 1)[0] for item in members)
    return TraceCollection(
        config=config,
        task=task,
        member_ids=members,
        bundle_ids=bundles,
        matrix=rows,
        case_ids=ids,
        cluster_ids=None,
        provenance=(str(provenance),),
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


def _pool_spec_fixed(member_count: int, name: str) -> PoolSpec:
    members = np.arange(member_count, dtype=np.int32)[None, :]
    local_pairs = np.asarray(
        list(itertools.combinations(range(member_count), 2)), dtype=np.int32
    )
    return PoolSpec(
        name=name,
        nominal_m=member_count,
        pool_members=members,
        pair_left=members[:, local_pairs[:, 0]],
        pair_right=members[:, local_pairs[:, 1]],
    )


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


def analyze_pool(
    collection: TraceCollection,
    spec: PoolSpec,
    *,
    b: int,
    seed: int,
    bootstrap_allowed: bool = True,
) -> dict[str, Any]:
    candidates = threshold_union(collection.matrix)
    point_rows = {
        item["value"]: pool_summary(collection.matrix, item["value"], spec)
        for item in candidates
    }
    supported_thresholds = [
        item["value"]
        for item in candidates
        if point_rows[item["value"]]["status"] == "support-valid"
    ]
    boot = (
        bootstrap_pool_rows(
            collection.matrix,
            supported_thresholds,
            spec,
            b=b,
            seed=seed,
            cluster_ids=collection.cluster_ids,
        )
        if supported_thresholds and bootstrap_allowed
        else {}
    )
    rows = [
        _threshold_record(item, point_rows[item["value"]], boot.get(item["value"]))
        for item in candidates
    ]
    p33_value = next(item["value"] for item in candidates if "P33" in item["sources"])
    primary = next(row for row in rows if row["threshold"]["value"] == p33_value)
    sensitivity = (
        next(
            row
            for row in reversed(rows)
            if row["status"] == "support-valid"
        )
        if supported_thresholds
        else None
    )
    selected = {
        "primary_P33": primary,
        "highest_supported_sensitivity": sensitivity,
    }
    return {
        "pool": spec.name,
        "K_bundle_count": len(set(collection.bundle_ids)),
        "M_operational_members": spec.nominal_m,
        "n_available_traces": len(collection.member_ids),
        "candidate_thresholds": candidates,
        "rows": rows,
        "selection": selected,
        "bootstrap": {
            "B": b,
            "seed": seed,
            "unit": (
                "subject cluster"
                if collection.cluster_ids is not None
                else "case"
                if bootstrap_allowed
                else "undefined: raw subject IDs unavailable"
            ),
        },
    }


def analyze_collection(
    collection: TraceCollection,
    *,
    b: int,
    seed: int,
    include_diverse4: bool,
    include_full: bool,
    fixed_subsets: Sequence[tuple[str, Sequence[int]]] = (),
    bootstrap_allowed: bool = True,
) -> dict[str, Any]:
    pools: list[dict[str, Any]] = []
    if include_diverse4:
        pools.append(
            analyze_pool(
                collection,
                _pool_spec_diverse4(collection.bundle_ids),
                b=b,
                seed=seed,
                bootstrap_allowed=bootstrap_allowed,
            )
        )
    if include_full:
        pools.append(
            analyze_pool(
                collection,
                _pool_spec_fixed(len(collection.member_ids), "full_trace_pool"),
                b=b,
                seed=seed + 10_000,
                bootstrap_allowed=bootstrap_allowed,
            )
        )
    for offset, (name, indices) in enumerate(fixed_subsets):
        submatrix = collection.matrix[np.asarray(indices)]
        subcollection = TraceCollection(
            config=collection.config,
            task=collection.task,
            member_ids=tuple(collection.member_ids[index] for index in indices),
            bundle_ids=tuple(collection.bundle_ids[index] for index in indices),
            matrix=submatrix,
            case_ids=collection.case_ids,
            cluster_ids=collection.cluster_ids,
            provenance=collection.provenance,
            evidence_scope=collection.evidence_scope,
            note=collection.note,
        )
        pools.append(
            analyze_pool(
                subcollection,
                _pool_spec_fixed(len(indices), name),
                b=b,
                seed=seed + 20_000 + offset,
                bootstrap_allowed=bootstrap_allowed,
            )
        )
    return {
        "config": collection.config,
        "task": collection.task,
        "status": "complete",
        "evidence_scope": collection.evidence_scope,
        "note": collection.note,
        "n_cases": len(collection.case_ids),
        "K_bundle_count": len(set(collection.bundle_ids)),
        "M_trace_count": len(collection.member_ids),
        "members": list(collection.member_ids),
        "source_files": list(collection.provenance),
        "pools": pools,
    }


def _format_value(item: dict[str, Any]) -> str:
    value = item["value"]
    if value is None:
        return "undefined"
    ci = item.get("ci95")
    if ci is None:
        return f"{value:.4f}"
    return f"{value:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"


def table_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Configuration-wise effective pool size",
        "",
        "| configuration | task | pool | threshold role | threshold | K | M | rho_fail | M_eff | support |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for config in payload["configurations"]:
        if not config["pools"]:
            lines.append(
                f"| {config['config']} | {config['task']} | unavailable | — | — | — | — | — | — | {config['status']}: {config['note']} |"
            )
            continue
        for pool in config["pools"]:
            primary = pool["selection"]["primary_P33"]
            sensitivity = pool["selection"]["highest_supported_sensitivity"]
            selected = [("P33 primary", primary)]
            if (
                sensitivity is not None
                and sensitivity["threshold"]["value"] != primary["threshold"]["value"]
            ):
                selected.append(("highest supported sensitivity", sensitivity))
            for role, row in selected:
                threshold = row["threshold"]["value"]
                if row["status"] == "support-valid":
                    values = (
                        _format_value(row["rho_fail"]),
                        _format_value(row["M_eff"]),
                        f"{row['support']['n_pools_supported']}/{row['support']['n_pools_total']} pools",
                    )
                else:
                    values = ("undefined", "undefined", UNDEFINED_REASON)
                lines.append(
                    f"| {config['config']} | {config['task']} | {pool['pool']} | {role} | "
                    f"{threshold:.6f} | {pool['K_bundle_count']} | {pool['M_operational_members']} | "
                    f"{values[0]} | {values[1]} | {values[2]} |"
                )
    lines.extend(
        [
            "",
            "All candidate-threshold rows, including undefined rows, are serialized in `metrics.json`.",
            "Scope: five primary frozen-encoder 1x1 configurations; controls use their separate scorers.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_config(path: Path, args: argparse.Namespace) -> None:
    text = (
        f"protocol: {Path(__file__).resolve()}\n"
        f"bootstrap_b: {args.bootstrap_b}\n"
        f"seed: {args.seed}\n"
        f"output_dir: {args.output_dir}\n"
        "thresholds: pooled_P25_P33_P50_union_fixed_0.70_0.80_0.85_0.90_0.95\n"
        "min_failures_per_member: 5\n"
        "min_passes_per_member: 5\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-b", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--traces-root",
        type=Path,
        required=True,
        help="primary trace root holding per_case_dice/",
    )
    parser.add_argument(
        "--stats-root",
        type=Path,
        default=None,
        help="dir holding m_eff/ (default: <traces-root>/stats)",
    )
    args = parser.parse_args()
    set_roots(args.traces_root, args.stats_root)
    if args.bootstrap_b <= 0:
        raise SystemExit("--bootstrap-b must be positive")

    output = args.output_dir
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    start = utc_now()
    start_monotonic = time.monotonic()
    meta = {
        "schema_version": "run_meta_v1",
        "status": "running",
        "started_at": start,
        "completed_at": None,
        "argv": sys.argv,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "pid": os.getpid(),
    }
    _write_config(output / "config.yaml", args)
    _json_dump(output / "meta.json", meta)
    log_path = logs / "run.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"{start} START {' '.join(sys.argv)}\n")
            log.flush()
            metrics = build_metrics(b=args.bootstrap_b, seed=args.seed)
            _json_dump(output / "metrics.json", metrics)
            (output / "effective_size_table.md").write_text(
                table_markdown(metrics), encoding="utf-8"
            )
            completed = utc_now()
            log.write(
                f"{completed} COMPLETE configurations={len(metrics['configurations'])} "
                f"elapsed_s={time.monotonic() - start_monotonic:.3f}\n"
            )
        meta.update(
            {
                "status": "complete",
                "completed_at": completed,
                "elapsed_s": time.monotonic() - start_monotonic,
                "outputs": ["config.yaml", "meta.json", "logs/run.log", "metrics.json", "effective_size_table.md"],
            }
        )
        _json_dump(output / "meta.json", meta)
    except Exception as error:
        meta.update(
            {
                "status": "failed",
                "completed_at": utc_now(),
                "elapsed_s": time.monotonic() - start_monotonic,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _json_dump(output / "meta.json", meta)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"{utc_now()} FAILED {type(error).__name__}: {error}\n")
        raise
    print(f"complete: {output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
