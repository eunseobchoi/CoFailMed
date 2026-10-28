"""Common-organ finalization, pinned mask evaluation and paired AMOS statistics."""
from __future__ import annotations
import hashlib, json, math, os, random, sys
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd
import nibabel as nib

EVALUATOR = Path(os.environ.get("COFAILMED_AMOS_EVALUATOR", "upstream/MRSegmentator/evaluation"))
BOOTSTRAP_B = 2000
PEARSON_SEED = 20260726
PHASE2_SEED = 0
SYSTEMS = ("mrseg_fold0", "mrseg_fold1", "totalseg")
ORGANS = (
    ("spleen", 1),
    ("kidney_right", 2),
    ("kidney_left", 3),
    ("gallbladder", 4),
    ("esophagus", 20),
    ("liver", 5),
    ("stomach", 6),
    ("aorta", 13),
    ("inferior_vena_cava", 14),
    ("pancreas", 7),
    ("adrenal_gland_right", 8),
    ("adrenal_gland_left", 9),
    ("duodenum", 22),
)
COMMON = (
    "spleen",
    "right_kidney",
    "left_kidney",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "inferior_vena_cava",
    "pancreas",
    "right_adrenal_gland",
    "left_adrenal_gland",
    "duodenum",
)
SAME = "same_system_mrseg_fold0_x_fold1"
CROSS0 = "cross_system_mrseg_fold0_x_totalseg"
CROSS1 = "cross_system_mrseg_fold1_x_totalseg"
CROSS = "cross_system_mean_of_two_pairs"
GAP = "gap_same_minus_cross_mean"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if len(x_array) != len(y_array) or len(x_array) < 2:
        raise ValueError("Pearson inputs require equal length >= 2")
    value = float(np.corrcoef(x_array, y_array)[0, 1])
    if not math.isfinite(value):
        raise RuntimeError("non-finite Pearson correlation")
    return value


def correlations(matrix: np.ndarray) -> dict[str, float]:
    fold0, fold1, total = matrix
    same = pearson(fold0, fold1)
    cross0 = pearson(fold0, total)
    cross1 = pearson(fold1, total)
    cross = (cross0 + cross1) / 2
    return {
        SAME: same,
        CROSS0: cross0,
        CROSS1: cross1,
        CROSS: cross,
        GAP: same - cross,
    }


def pearson_bootstrap(
    matrix: np.ndarray,
    rng: random.Random,
) -> dict[str, Any]:
    point = correlations(matrix)
    draws = {key: [] for key in point}
    n = matrix.shape[1]
    for _ in range(BOOTSTRAP_B):
        indices = [rng.randrange(n) for _ in range(n)]
        values = correlations(matrix[:, indices])
        for key, value in values.items():
            draws[key].append(value)
    valid = len(draws[GAP])
    if valid != BOOTSTRAP_B:
        raise RuntimeError(
            f"locked Pearson bootstrap requires {BOOTSTRAP_B}/{BOOTSTRAP_B} "
            f"finite replicates, got {valid}/{BOOTSTRAP_B}"
        )
    return {
        "pearson_r": point,
        "bootstrap": {
            "requested_replicates": BOOTSTRAP_B,
            "valid_replicates": valid,
            "degenerate_replicates": 0,
            "seed": PEARSON_SEED,
            "unit": "one shared case-index vector within modality",
            "percentile_method": "linear",
            "percentile_ci95": {
                key: [percentile(values, 0.025), percentile(values, 0.975)]
                for key, values in draws.items()
            },
        },
    }


def validate_nifti(path: Path, reference: Path, *, binary: bool) -> None:
    image = nib.load(path)
    ref = nib.load(reference)
    if image.shape != ref.shape or not np.allclose(image.affine, ref.affine, atol=1e-3):
        raise RuntimeError(f"grid mismatch: {path}")
    data = np.asanyarray(image.dataobj)
    if not np.isfinite(data).all() or not np.equal(data, np.rint(data)).all():
        raise RuntimeError(f"invalid segmentation values: {path}")
    unique = set(int(value) for value in np.unique(data))
    allowed = {0, 1} if binary else set(range(41))
    if not unique.issubset(allowed):
        raise RuntimeError(f"label set mismatch: {path}")
    if not binary and not any(value > 0 for value in unique):
        raise RuntimeError(f"empty MRSegmentator prediction: {path}")


def phase2_summaries(
    matrix: np.ndarray,
    case_ids: tuple[str, ...],
    *,
    modality: str,
) -> dict[str, Any]:
    from analysis.effective_size import (
        TraceCollection,
        _pool_spec_fixed,
        analyze_pool,
    )
    from analysis.failure_comparison import (
        METRICS,
        UNDEFINED_REASON,
        _floor_gate,
        _support,
        amos_pair_masks,
        bootstrap_metrics,
        metric_points,
    )

    collection = TraceCollection(
        config="amos22_fixed_system_instances_scaleup",
        task=f"amos22_{modality}",
        member_ids=SYSTEMS,
        bundle_ids=SYSTEMS,
        matrix=matrix,
        case_ids=case_ids,
        cluster_ids=None,
        provenance=(),
        evidence_scope="fixed 60-case sample per modality",
        note="All three instances are nnU-Net-family.",
    )
    meff = analyze_pool(
        collection,
        _pool_spec_fixed(3, "full_trace_pool"),
        b=BOOTSTRAP_B,
        seed=PHASE2_SEED,
    )
    primary = meff["selection"]["primary_P33"]
    threshold = float(primary["threshold"]["value"])
    within_pairs, cross_pairs = amos_pair_masks(SYSTEMS)
    functional_floor = _floor_gate(collection)
    support = _support(matrix, threshold)
    if support["status"] != "support-valid":
        nonpearson = {
            "status": UNDEFINED_REASON,
            "functional_floor": functional_floor,
            "threshold": {
                "value": threshold,
                "sources": primary["threshold"]["sources"],
            },
            "support": support,
            "estimators": {
                metric: {"status": UNDEFINED_REASON, "point": None, "ci95": None}
                for metric in METRICS
            },
        }
    else:
        points = metric_points(matrix >= threshold, within_pairs, cross_pairs)
        intervals = bootstrap_metrics(
            matrix,
            threshold,
            within_pairs,
            cross_pairs,
            b=BOOTSTRAP_B,
            seed=PHASE2_SEED,
        )
        nonpearson = {
            "status": "complete",
            "functional_floor": functional_floor,
            "threshold": {
                "value": threshold,
                "sources": primary["threshold"]["sources"],
            },
            "support": support,
            "pairing": {"same_pairs": 1, "cross_pairs": 2},
            "estimators": {
                metric: {
                    "status": "complete",
                    "point": points[metric],
                    "case_bootstrap_95ci": intervals[metric],
                }
                for metric in METRICS
            },
        }
    return {
        "M_eff": meff,
        "nonpearson_P33": nonpearson,
    }


def merge_organ_masks(directory: Path, reference_path: Path, output: Path) -> None:
    """Merge the common-organ binary masks in the recorded order and label map."""
    reference = nib.load(reference_path)
    volume = np.zeros(reference.shape, dtype=np.uint8)
    for organ, label_id in ORGANS:
        mask_path = directory / f"{organ}.nii.gz"
        validate_nifti(mask_path, reference_path, binary=True)
        mask = np.asanyarray(nib.load(mask_path).dataobj)
        volume[mask > 0.5] = label_id
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(volume, reference.affine, header), output)


def shared_missingness_macros(
    matrices: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    reference = None
    macros = {}
    for system in SYSTEMS:
        values = np.asarray(matrices[system], dtype=float)
        if values.shape != (120, 13) or np.isinf(values).any():
            raise RuntimeError(f"invalid 120x13 organ Dice matrix: {system}")
        finite = np.isfinite(values)
        if reference is None:
            reference = finite
        elif not np.array_equal(finite, reference):
            raise RuntimeError("organ Dice finite/missing masks differ across systems")
        available = finite.sum(axis=1)
        if np.any(available < 1):
            raise RuntimeError("a case has no finite common-organ Dice values")
        macros[system] = np.nanmean(values, axis=1)
        if not np.isfinite(macros[system]).all():
            raise RuntimeError(f"non-finite skipna macro Dice: {system}")
    assert reference is not None
    missing_by_organ = {
        organ: int((~reference[:, index]).sum())
        for index, organ in enumerate(COMMON)
        if int((~reference[:, index]).sum()) > 0
    }
    return macros, {
        "status": "PASS",
        "convention": "mean(axis=1, skipna=True)",
        "systems_share_identical_finite_missing_mask": True,
        "undefined_case_organ_cells_per_system": int((~reference).sum()),
        "finite_case_organ_cells_per_system": int(reference.sum()),
        "minimum_finite_organs_per_case": int(reference.sum(axis=1).min()),
        "maximum_finite_organs_per_case": int(reference.sum(axis=1).max()),
        "missing_by_organ": missing_by_organ,
        "all_120_macros_finite": True,
    }


def finalize_organ_frames(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Align the fixed cohort, require shared missingness, then average finite organs."""
    if set(frames) != set(SYSTEMS):
        raise ValueError("expected all three systems")
    matrices = {}
    identity = None
    for system in SYSTEMS:
        frame = frames[system].sort_values("case").reset_index(drop=True)
        if frame["case"].isna().any() or frame["case"].astype(str).str.len().eq(0).any():
            raise ValueError("case IDs must be nonempty")
        if frame["case"].duplicated().any():
            raise ValueError("duplicate case IDs")
        if len(frame) != 120 or frame.groupby("modality").size().to_dict() != {"ct": 60, "mri": 60}:
            raise ValueError("expected 60 CT and 60 MRI cases")
        current = frame[["case", "modality"]]
        if identity is None:
            identity = current
        elif not current.equals(identity):
            raise ValueError("case/modality identities differ across systems")
        values = frame[list(COMMON)].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if np.any((finite < 0) | (finite > 1)):
            raise ValueError("organ Dice must be within [0, 1]")
        matrices[system] = values
    macros, missingness = shared_missingness_macros(matrices)
    pivot = identity.copy()
    for system in SYSTEMS:
        pivot[system] = macros[system]
    return pivot.sort_values(["modality", "case"]).reset_index(drop=True), missingness


def run_evaluator(
    evaluation: str,
    labels: str,
    segmentation: str,
    system: str,
) -> str:
    predictions = sorted(Path(segmentation).glob("*.nii.gz"))
    if not predictions:
        raise ValueError("no prediction NIfTI files")
    for prediction in predictions:
        target = Path(labels) / prediction.name.replace("_seg", "")
        pred, truth = nib.load(prediction), nib.load(target)
        if len(pred.shape) != 3 or pred.shape != truth.shape or not np.allclose(
            pred.affine, truth.affine, atol=1e-3, rtol=0
        ):
            raise ValueError(f"prediction/target physical grid mismatch: {prediction.name}")
        for volume, maximum in ((pred, 40), (truth, 15)):
            values = np.asanyarray(volume.dataobj)
            if (not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all()
                    or np.any((values < 0) | (values > maximum))):
                raise ValueError(f"invalid integer label values: {prediction.name}")
    # Cap evaluator threads when systems are scored in parallel.
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "2"
    # The published acquisition command pins these two unchanged upstream files.
    evaluator = EVALUATOR.resolve()
    expected = {
        "eval_data.py": "0a1abf4aea34e25e93806accd520aaecf4d92ac142f1eafdc1b2c917c997dd9c",
        "mappings.py": "818647f5e8ce808b601e1bdb10f174db4cf022dbbe15ef4bd128deaa23257c22",
    }
    for name, digest in expected.items():
        if sha256(evaluator / name) != digest:
            raise ValueError(f"upstream evaluator hash mismatch: {name}")
    import importlib.metadata
    if importlib.metadata.version("monai") != "1.5.0":
        raise ValueError("this evaluator requires the recorded MONAI 1.5.0")
    sys.path.insert(0, str(evaluator))
    import eval_data
    import mappings
    if Path(eval_data.__file__).resolve() != evaluator / "eval_data.py" or Path(mappings.__file__).resolve() != evaluator / "mappings.py":
        raise ValueError("loaded evaluator modules differ from the pinned source directory")

    old = Path.cwd()
    os.chdir(evaluation)
    try:
        eval_data.main(
            labels,
            segmentation,
            system,
            mappings.amos_to_mrseg,
            (1.5, 1.5, 3.0),
        )
    finally:
        os.chdir(old)
    return system


def aggregate_metrics(pivot: pd.DataFrame) -> dict[str, Any]:
    by_modality = {}
    # As in the original finalizer: one RNG is initialized once, then consumed
    # by CT followed by MRI (pivot is sorted modality/case).
    rng = random.Random(PEARSON_SEED)
    for modality in ("ct", "mri"):
        selected = pivot[pivot["modality"] == modality]
        matrix = selected[list(SYSTEMS)].to_numpy(dtype=float).T
        primary = pearson_bootstrap(matrix, rng)
        primary["n"] = len(selected)
        primary["case_order"] = selected["case"].astype(str).tolist()
        primary["mean_macro_dice_13"] = {
            system: float(selected[system].mean()) for system in SYSTEMS
        }
        primary["phase2_secondary"] = phase2_summaries(
            matrix,
            tuple(selected["case"].astype(str)),
            modality=modality,
        )
        by_modality[modality] = primary
    return by_modality
