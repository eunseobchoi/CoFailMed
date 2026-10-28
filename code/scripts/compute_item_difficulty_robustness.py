#!/usr/bin/env python3
"""CPU-only item-difficulty robustness diagnostics.

This script uses only released per-case Dice JSONs. It computes two artifact
checks that do not require raw images, feature caches, checkpoints, or GPUs:

1. Functional-floor grid over the prespecified floor values.
2. Exploratory disjoint-family panel residualization. All evaluation traces
   in a panel share one reference-family mean nuisance regressor. Overlapping
   panels define a finite-panel sensitivity, not independent replications.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
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

from fmpool.estimators import _bootstrap_case_indices, functional_floor_filter, within_cross_rbar  # noqa: E402

import compute_all as ca  # noqa: E402


SCHEMA_VERSION = "fmpool_item_difficulty_robustness_v2"
TASKS = ca.TASKS
DEFAULT_FLOORS = (0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
PRIMARY_FLOOR = 0.30
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0

BROAD_FAMILY = {
    "dinov2_vitb14": "dinov2",
    "dinov2_vits14": "dinov2",
    "dinov2_vitl14": "dinov2",
    "dinov2_vitg14": "dinov2",
    "clip_vitb16": "openai_clip",
    "clip_vitb32": "openai_clip",
    "clip_vitl14": "openai_clip",
    "resnet18": "resnet",
    "resnet34": "resnet",
    "resnet50": "resnet",
    "resnet101": "resnet",
    "convnext_tiny": "convnext",
    "convnext_small": "convnext",
    "convnext_base": "convnext",
    "efficientnet_b0": "efficientnet",
    "efficientnet_b1": "efficientnet",
    "efficientnet_b2": "efficientnet",
    "efficientnet_b3": "efficientnet",
    "mae_vitb16": "mae",
    "mae_vitl16": "mae",
    "deit_vitb16": "deit",
    "deit_vits16": "deit",
    "deit_vitt16": "deit",
    "biomedclip": "biomedclip",
    "retfound_vitl16": "retfound",
    "medsam_vitb16": "medsam",
}


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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_json_value(payload), fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")


def _ci95(values: list[float]) -> list[float | None]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return [None, None]
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return [float(lo), float(hi)]


def _fm_mean(per_case: dict[tuple[str, int], np.ndarray]) -> dict[str, float]:
    by_fm: dict[str, list[np.ndarray]] = {}
    for (fm, _seed), arr in per_case.items():
        by_fm.setdefault(fm, []).append(np.asarray(arr, dtype=np.float64))
    return {
        fm: float(np.mean([float(np.mean(arr)) for arr in arrs]))
        for fm, arrs in by_fm.items()
    }


def _point(per_case: dict[tuple[str, int], np.ndarray]) -> dict[str, Any]:
    stats = within_cross_rbar(
        per_case,
        family_pairs_within=ca.DINOV2_FAMILY_PAIRS,
        stat="pearson",
    )
    return {
        "within_rbar": float(stats["within"]),
        "cross_rbar": float(stats["cross"]),
        "gap": float(stats["gap"]),
        "n_within_pairs": int(stats["n_within_pairs"]),
        "n_cross_pairs": int(stats["n_cross_pairs"]),
    }


def _family(fm: str) -> str:
    return BROAD_FAMILY.get(fm, fm)


def _residualize_one(y: np.ndarray, difficulty: np.ndarray) -> tuple[np.ndarray, dict]:
    y = np.asarray(y, dtype=np.float64)
    difficulty = np.asarray(difficulty, dtype=np.float64)
    if y.shape != difficulty.shape:
        raise ValueError(f"residualization shape mismatch: {y.shape} vs {difficulty.shape}")
    if y.size < 2 or float(np.std(difficulty)) < 1e-12:
        residual = y - float(np.mean(y))
        return residual, {"intercept": float(np.mean(y)), "slope": 0.0, "r2": 0.0}
    x = np.column_stack([np.ones_like(difficulty), difficulty])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residual = y - fitted
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - fitted) ** 2))
    r2 = 0.0 if ss_tot <= 1e-24 else max(0.0, 1.0 - ss_res / ss_tot)
    return residual, {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "r2": float(r2),
    }


def _enumerated_panel_estimates(
    per_case: dict[tuple[str, int], np.ndarray],
) -> list[dict[str, Any]]:
    """All floor(F/2)-family reference subsets; complementary evaluation panels.

    The nuisance regressor averages traces within each reference broad family,
    then averages those family means equally. Evaluation traces never contribute
    to it. The original within/cross pair rule is retained inside each panel.
    """
    families = sorted({_family(fm) for fm, _ in per_case})
    if len(families) < 3:
        raise ValueError("need at least three broad families for disjoint panels")
    arrays = {k: np.asarray(v, dtype=float) for k, v in per_case.items()}
    data = np.stack(list(arrays.values()))
    if data.ndim != 2 or data.shape[1] < 2 or not np.isfinite(data).all():
        raise ValueError("expected aligned finite score traces with at least two cases")
    family_means = {f: np.mean([v for (fm, _), v in arrays.items() if _family(fm) == f], axis=0)
                    for f in families}
    rows = []
    for reference in combinations(families, len(families) // 2):
        difficulty = np.mean([family_means[f] for f in reference], axis=0)
        evaluation = {k: arrays[k] for k in sorted(arrays) if _family(k[0]) not in reference}
        # One shared design matrix and multi-response fit, not family-specific leaveout.
        values = np.stack(list(evaluation.values()), axis=1)
        if np.std(difficulty) < 1e-12:
            residuals = values - values.mean(axis=0)
        else:
            design = np.column_stack([np.ones_like(difficulty), difficulty])
            residuals = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
        residual = dict(zip(evaluation, residuals.T))
        raw_point, residual_point = _point(evaluation), _point(residual)
        rows.append({
            "reference_families": list(reference),
            "evaluation_families": sorted(set(families) - set(reference)),
            "n_reference_traces": len(arrays) - len(evaluation),
            "n_evaluation_traces": len(evaluation),
            "raw": raw_point,
            "residualized": residual_point,
            "eligible": math.isfinite(raw_point["gap"]) and math.isfinite(residual_point["gap"]),
        })
    return rows


def _panel_mean(rows: list[dict], field: str) -> dict[str, float]:
    # An undefined panel does not silently redefine the equally weighted target.
    return {key: float(np.mean([row[field][key] for row in rows]))
            for key in ("within_rbar", "cross_rbar", "gap")}


def _attenuation(raw_gap: float, residual_gap: float) -> float:
    if not math.isfinite(raw_gap) or abs(raw_gap) < 1e-12:
        return float("nan")
    return float(residual_gap / raw_gap)


def _item_difficulty_analysis(
    per_case: dict[tuple[str, int], np.ndarray],
    *,
    n_boot: int,
    seed: int,
    case_to_subject: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    panels = _enumerated_panel_estimates(per_case)
    raw, residual = _panel_mean(panels, "raw"), _panel_mean(panels, "residualized")
    raw_gap, residual_gap = raw["gap"], residual["gap"]
    boot_raw, boot_residual, boot_difference, boot_attenuation = [], [], [], []
    n = len(next(iter(per_case.values())))
    for idx in _bootstrap_case_indices(n, n_boot, seed, case_to_subject):
        rows = _enumerated_panel_estimates({k: np.asarray(arr)[idx] for k, arr in per_case.items()})
        a, b = _panel_mean(rows, "raw")["gap"], _panel_mean(rows, "residualized")["gap"]
        if math.isfinite(a):
            boot_raw.append(a)
        if math.isfinite(b):
            boot_residual.append(b)
        if math.isfinite(a) and math.isfinite(b):
            boot_difference.append(b - a)
            att = _attenuation(a, b)
            if math.isfinite(att):
                boot_attenuation.append(att)
    gaps = [p["residualized"]["gap"] for p in panels if p["eligible"]]
    return {
        "description": "Common nuisance regressor from disjoint reference broad families in each panel.",
        "target": "equal-panel mean of within-minus-cross correlations in complementary evaluation panels",
        "interpretation": "exploratory, noncausal finite-panel sensitivity; panels overlap and are not independent",
        "conditioning": "observed models, floor-selected pool and panel rule; selection uncertainty excluded",
        "difficulty_family_rule": "equal reference-family means; exhaustive floor(F/2) reference subsets",
        "within_cross_family_rule": "paper primary rule: same FM plus DINOv2-B/S pair",
        "raw_full_pool": _point(per_case),
        "raw": raw,
        "residualized": residual,
        "residualized_gap_minus_raw_gap": residual_gap - raw_gap,
        "attenuation_ratio_residual_over_raw": _attenuation(raw_gap, residual_gap),
        "panels": panels,
        "panel_spread": {
            "kind": "descriptive spread among eligible panels, not uncertainty on their mean",
            "n_panels": len(panels), "n_eligible": len(gaps),
            "range": [min(gaps), max(gaps)] if gaps else [None, None],
            "sd": float(np.std(gaps)) if gaps else None,
            "positive": sum(g > 1e-12 for g in gaps),
            "ties_at_1e-12": sum(abs(g) <= 1e-12 for g in gaps),
            "negative": sum(g < -1e-12 for g in gaps),
        },
        "bootstrap": {
            "B": n_boot, "seed": seed,
            "unit": "subject" if case_to_subject is not None else "case",
            "target": "conditional on retained pool; refit common nuisance on every paired draw",
            "B_completed_raw": len(boot_raw), "B_completed_residual": len(boot_residual),
            "B_completed_pair": len(boot_difference), "B_completed_ratio": len(boot_attenuation),
            "raw_gap_95ci": _ci95(boot_raw),
            "residualized_gap_95ci": _ci95(boot_residual),
            "residualized_minus_raw_gap_95ci": _ci95(boot_difference),
            "attenuation_ratio_95ci": _ci95(boot_attenuation),
            "ratio_caution": "unstable near a zero raw gap; use the paired difference as the effect",
        },
    }


def _floor_grid(
    per_case_all: dict[tuple[str, int], np.ndarray],
    *,
    unit_ids: tuple[str, ...],
    floors: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    has_subject_units = len(set(unit_ids)) < len(unit_ids)
    for floor in floors:
        if floor <= 0:
            per_case = dict(per_case_all)
        else:
            per_case = functional_floor_filter(
                per_case_all,
                min_mean_dice=float(floor),
                scope="per_fm",
            )
        fms = sorted({fm for fm, _seed in per_case})
        row: dict[str, Any] = {
            "floor": float(floor),
            "n_fms": int(len(fms)),
            "n_members": int(len(per_case)),
            "fms_kept": fms,
            "fm_mean_dice": _fm_mean(per_case),
            "case_level": _point(per_case) if len(per_case) >= 2 else None,
        }
        if has_subject_units and len(per_case) >= 2:
            aggregated, units = ca._aggregate_by_unit(per_case, unit_ids)
            row["subject_level"] = {
                "n_units": int(len(units)),
                **_point(aggregated),
            }
        rows.append(row)
    return rows


def _task_analysis(
    results_root: Path,
    task: str,
    *,
    floors: tuple[float, ...],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    fm_jsons, _nnunet_jsons = ca._collect_task_jsons(results_root, task)
    if not fm_jsons:
        raise FileNotFoundError(f"no per-case Dice JSONs found for task {task!r}")
    ca._validate_alignment(fm_jsons)
    per_case_all = ca._as_per_case_map(fm_jsons)
    meta = ca._case_metadata(task, fm_jsons[0].test_ids)
    unit_ids = tuple(str(x) for x in meta["unit_ids"])

    primary = functional_floor_filter(
        per_case_all,
        min_mean_dice=PRIMARY_FLOOR,
        scope="per_fm",
    )
    out: dict[str, Any] = {
        "task": task,
        "n_test": int(len(fm_jsons[0].test_ids)),
        "case_unit": meta["case_unit"],
        "cluster_unit": meta["cluster_unit"],
        "n_units": int(meta["n_units"]),
        "source_files": ca._rel_paths([j.path for j in fm_jsons], results_root),
        "available_fms": sorted({fm for fm, _seed in per_case_all}),
        "available_members": int(len(per_case_all)),
        "floor_grid": _floor_grid(
            per_case_all,
            unit_ids=unit_ids,
            floors=floors,
        ),
        "disjoint_panel_item_difficulty_floor_0.30": _item_difficulty_analysis(
            primary,
            n_boot=n_boot,
            seed=seed,
            case_to_subject=unit_ids if meta["cluster_unit"] is not None else None,
        ),
    }
    if meta["cluster_unit"] is not None:
        aggregated, units = ca._aggregate_by_unit(primary, unit_ids)
        out["subject_level_disjoint_panel_item_difficulty_floor_0.30"] = {
            "n_units": int(len(units)),
            **_item_difficulty_analysis(
                aggregated,
                n_boot=n_boot,
                seed=seed,
            ),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True,
                        help="merged results root (contains per_case_dice/)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--floors", nargs="*", type=float, default=list(DEFAULT_FLOORS))
    parser.add_argument("--n-boot", type=int, default=BOOTSTRAP_B)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = args.results_root
    floors = tuple(float(x) for x in args.floors)
    task_rows = {
        task: _task_analysis(
            results_root,
            task,
            floors=floors,
            n_boot=int(args.n_boot),
            seed=int(args.seed),
        )
        for task in args.tasks
    }
    out = {
        "schema_version": SCHEMA_VERSION,
        "rig_version": ca.STATISTICS_RIG_VERSION,
        "generated_at": _now(),
        "results_root": str(results_root),
        "primary_floor": PRIMARY_FLOOR,
        "functional_floor_grid": [float(x) for x in floors],
        "bootstrap_B": int(args.n_boot),
        "bootstrap_seed": int(args.seed),
        "broad_family_rule": BROAD_FAMILY,
        "task_rows": task_rows,
    }
    _write_json(args.out, out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
