#!/usr/bin/env python3
"""Guarded nnU-Net v2.6.2 cell: pinned inputs, private workspace, 1e-6-smoothed Dice scoring.

Rescore with rescore_nnunet_primary_dice.py for the paper's unsmoothed primary Dice.

Requires a fresh preprocessing receipt pinned by input_manifest_v2 through
nnunet_preprocessing_sha256. Never resumes an existing output/cell or modifies
source datasets, preprocessed workspaces, or the installed discovery shim.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import traceback
from importlib.metadata import version
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from fmpool import provenance
from fmpool import determinism

logger = logging.getLogger("fmpool.train_nnunet_baseline")

_POSITIVE_LABEL = {"kvasir": 1, "acdc_lv": 1, "brats_wt": None, "riga_cup": 1, "riga_disc": 1}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a seeded nnU-Net v2 baseline."
    )
    p.add_argument("--task", required=True, choices=["riga_cup", "acdc_lv"])
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--dataset-id",
        type=int,
        required=True,
        help="nnU-Net raw-dataset numeric ID (e.g. 27 for Dataset027).",
    )
    p.add_argument(
        "--config",
        default="3d_fullres",
        choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
    )
    p.add_argument(
        "--base-trainer",
        default="nnUNetTrainer_100epochs",
        help="Base upstream trainer to seed. The paper uses 100 epochs.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for per-case Dice JSON.",
    )
    p.add_argument(
        "--nnunet-raw",
        type=str,
        default=None,
        help="Override nnUNet_raw root; otherwise uses env var.",
    )
    p.add_argument("--nnunet-preprocessed", type=str, default=None)
    p.add_argument("--nnunet-results", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--input-manifest", type=Path, required=True)
    p.add_argument("--preprocessing-evidence", type=Path, required=True)
    p.add_argument("--out-json", type=Path)
    p.add_argument("--evaluation", choices=["fold", "holdout"], default="fold")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# nnU-Net env wiring
# ---------------------------------------------------------------------------


def _set_nnunet_env(args: argparse.Namespace) -> None:
    """Propagate CLI overrides into the nnU-Net environment variables."""
    if args.nnunet_raw:
        os.environ["nnUNet_raw"] = str(args.nnunet_raw)
    if args.nnunet_preprocessed:
        os.environ["nnUNet_preprocessed"] = str(args.nnunet_preprocessed)
    if args.nnunet_results:
        os.environ["nnUNet_results"] = str(args.nnunet_results)


def _locate_raw_dataset(dataset_id: int) -> Path | None:
    """Find ``nnUNet_raw/Dataset{ID:03d}_*`` on disk (if already staged)."""
    root_env = os.environ.get("nnUNet_raw")
    if not root_env:
        return None
    root = Path(root_env)
    if not root.is_dir():
        return None
    prefix = f"Dataset{dataset_id:03d}_"
    matches = sorted(root.glob(prefix + "*"))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("ambiguous nnUNet raw dataset ID")
    return matches[0]


# ---------------------------------------------------------------------------
# Per-case Dice from predicted vs GT NIfTI
# ---------------------------------------------------------------------------


def _binary_dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    tgt_b = target.astype(bool)
    p_sum = float(pred_b.sum())
    t_sum = float(tgt_b.sum())
    if p_sum == 0.0 and t_sum == 0.0:
        return 1.0
    inter = float(np.logical_and(pred_b, tgt_b).sum())
    return (2.0 * inter) / (p_sum + t_sum + 1e-6)


def _load_nifti_volume(path: Path) -> np.ndarray:
    """Read a NIfTI volume via SimpleITK, falling back to nibabel."""
    import nibabel as nib

    native = np.asarray(nib.load(str(path)).dataobj)
    # SimpleITK's NIfTI reader replaces NaN/Inf voxels with zero.
    if not np.isfinite(native).all():
        raise ValueError(f"nonfinite NIfTI voxels: {path.name}")
    try:
        import SimpleITK as sitk  # type: ignore

        img = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(img)
    except ImportError:
        return native


def _positive_label_from_dataset_json(path: Path) -> int:
    labels = json.loads(path.read_text())["labels"]
    if labels.get("background") != 0 or sorted(labels.values()) != [0, 1]:
        raise ValueError("expected exported binary dataset labels {0, 1}")
    return 1


def _validate_masks(pred, gt, positive_label, name):
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError(f"nonfinite prediction/GT: {name}")
    if pred.shape != gt.shape:
        raise ValueError(f"prediction/GT shape mismatch: {name}")
    if positive_label == 1 and (not np.isin(pred, [0, 1]).all() or not np.isin(gt, [0, 1]).all()):
        raise ValueError(f"expected exported binary label palette: {name}")


def _validate_nifti_pair(image_path, reference_path, name):
    import nibabel as nib
    import SimpleITK as sitk

    for path in (image_path, reference_path):
        volume = nib.load(path)
        spacing = np.asarray(volume.header.get_zooms())
        if len(volume.shape) != 3:
            raise ValueError(f"non-3D NIfTI volume: {name}")
        if (not np.isfinite(volume.affine).all() or not np.isfinite(spacing).all()
                or (spacing <= 0).any()):
            raise ValueError(f"nonfinite or degenerate NIfTI geometry: {name}")
    # nnU-Net SimpleITKIO uses this grid, not nibabel's selected sform alone.
    image, reference = (sitk.ReadImage(str(path)) for path in (image_path, reference_path))
    if image.GetSize() != reference.GetSize():
        raise ValueError(f"image/reference shape mismatch: {name}")
    for field in ("GetSpacing", "GetOrigin", "GetDirection"):
        actual, expected = np.asarray(getattr(image, field)()), np.asarray(getattr(reference, field)())
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError(f"nonfinite NIfTI geometry: {name}")
        if not np.allclose(actual, expected, rtol=0, atol=1e-5):
            raise ValueError(f"image/reference geometry mismatch ({field}): {name}")
    return image, reference


def _collect_per_case_dice(
    pred_dir: Path,
    gt_dir: Path,
    positive_label: int | None = None,
    *, expected_ids: list[str] | None = None,
) -> tuple[list[str], list[float]]:
    """Pair predicted vs GT NIfTI by filename and compute per-case Dice.

    ``positive_label`` selects a single foreground class (
    exported binary tasks use ``label==1``). When ``None`` the Dice is over all
    non-zero voxels (BraTS foreground derivative).
    """
    predictions = _label_files(pred_dir)
    ground_truth = _label_files(gt_dir)
    if expected_ids is None:
        # Standalone metric compatibility only; partitioned datasets need a fold.
        if gt_dir.name == "labelsTr":
            raise ValueError("expected fold IDs required for labelsTr")
        expected_ids = sorted(ground_truth)
    expected_ids = _unique_ids(expected_ids, "expected scoring IDs")
    _same_ids(predictions, expected_ids, "prediction IDs")
    if not set(expected_ids) <= ground_truth.keys():
        raise ValueError("missing expected GT IDs")
    pred_files = [predictions[cid] for cid in expected_ids]
    ids: list[str] = []
    scores: list[float] = []
    for pred_path in pred_files:
        name = pred_path.name
        gt_path = gt_dir / name
        if not gt_path.is_file():
            raise FileNotFoundError(f"missing GT for {name}")
        if name.endswith(".png"):
            from PIL import Image
            pred, gt = np.asarray(Image.open(pred_path)), np.asarray(Image.open(gt_path))
        else:
            pred = _load_nifti_volume(pred_path)
            gt = _load_nifti_volume(gt_path)
        _validate_masks(pred, gt, positive_label, name)
        if positive_label is not None:
            pred_bin = pred == positive_label
            gt_bin = gt == positive_label
        else:
            pred_bin = pred > 0
            gt_bin = gt > 0
        ids.append(name.removesuffix(".nii.gz").removesuffix(".png"))
        scores.append(_binary_dice(pred_bin, gt_bin))
    return ids, scores


# ---------------------------------------------------------------------------
# Atomic JSON
# ---------------------------------------------------------------------------


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"preserving existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".tmp.", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.link(tmp_name, path)
        os.unlink(tmp_name)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _unique_ids(ids, label):
    if not isinstance(ids, list) or not ids or any(
        not isinstance(cid, str) or not cid or "/" in cid or "\\" in cid or cid in {".", ".."}
        for cid in ids
    ) or len(ids) != len(set(ids)):
        raise ValueError(f"invalid/empty/duplicate {label}")
    return ids


def _same_ids(actual, expected, label):
    if set(actual) != set(expected):
        raise ValueError(f"{label}: missing={sorted(set(expected) - set(actual))}, "
                         f"extra={sorted(set(actual) - set(expected))}")


def _label_files(folder):
    files = {}
    for path in sorted([*folder.glob("*.nii.gz"), *folder.glob("*.png")]):
        cid = path.name.removesuffix(".nii.gz").removesuffix(".png")
        if cid in files:
            raise ValueError(f"duplicate label ID: {cid}")
        files[cid] = path
    return files


def _tree_hashes(root):
    root = Path(root).absolute()
    if not root.is_dir() or root.resolve() != root:
        raise ValueError(f"missing/symlinked input tree: {root}")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlinked input rejected: {path}")
        if path.is_file():
            files[str(path.relative_to(root))] = determinism.sha256_file(path)
    if not files:
        raise ValueError(f"empty input tree: {root}")
    return files


def _upstream_identity():
    import nnunetv2
    installed = version("nnunetv2")
    if installed != "2.6.2":
        raise ValueError(f"reviewed nnunetv2 2.6.2 required, got {installed}")
    root = Path(nnunetv2.__path__[0])
    return {"version": installed, "source_sha256": {
        str(p.relative_to(root)): determinism.sha256_file(p) for p in sorted(root.rglob("*.py"))}}


def _write_json(path, value):
    _atomic_write_bytes(Path(path), json.dumps(value, indent=2, allow_nan=False).encode())


def _failure(path, exc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".failed.", suffix=".json", dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        json.dump({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}, stream)
    logger.error("preserved failure: %s", name)


def prepare_preprocessing(raw, preprocessed_root, receipt_path):
    """Fresh upstream preprocessing; never bless or overwrite an existing cache."""
    raw, preprocessed_root, receipt_path = map(lambda p: Path(p).absolute(),
                                              (raw, preprocessed_root, receipt_path))
    work = receipt_path.with_suffix(".workspace")
    if receipt_path.exists() or work.exists() or (preprocessed_root / raw.name).exists():
        raise FileExistsError("preprocessing requires fresh receipt/workspace/dataset paths")
    sys.dont_write_bytecode = True
    source = _upstream_identity()
    raw_hashes = _tree_hashes(raw)
    work.mkdir(parents=True, exist_ok=False)
    copied_raw = work / "raw" / raw.name
    shutil.copytree(raw, copied_raw)
    if _tree_hashes(copied_raw) != raw_hashes:
        raise ValueError("raw copy changed during preprocessing staging")
    env = dict(os.environ, nnUNet_raw=str(copied_raw.parent),
               nnUNet_preprocessed=str(preprocessed_root), nnUNet_results=str(work / "results"),
               PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-c",
               "from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry; plan_and_preprocess_entry()",
               "-d", str(int(raw.name[7:10])),
               "--verify_dataset_integrity"]
    try:
        subprocess.run(command, env=env, check=True)
        pre = preprocessed_root / raw.name
        if _tree_hashes(raw) != raw_hashes or _tree_hashes(copied_raw) != raw_hashes:
            raise ValueError("raw export changed during preprocessing")
        if json.loads((pre / "splits_final.json").read_text()) != json.loads((raw / "splits_final.json").read_text()):
            raise ValueError("planner did not preserve declared splits")
        _write_json(receipt_path, {"schema_version": "nnunet_preprocessing_v2",
            "dataset": raw.name, "raw_files": raw_hashes, "preprocessed_files": _tree_hashes(pre),
            "upstream": source, "runtime": provenance.runtime_identity(), "command": command,
            "producer_source": Path(__file__).read_text(),
            "raw_source": str(raw), "raw_consumed": str(copied_raw), "preprocessed": str(pre)})
    except Exception as exc:
        _failure(receipt_path, exc)
        raise


def validate_workspace(args, inputs):
    raw = _locate_raw_dataset(args.dataset_id)
    if raw is None:
        raise FileNotFoundError("stage and pin a fresh raw export before launch")
    raw = raw.absolute()
    pre = Path(os.environ["nnUNet_preprocessed"]).absolute() / raw.name
    receipt_sha = determinism.sha256_file(args.preprocessing_evidence)
    if receipt_sha != inputs["manifest"].get("nnunet_preprocessing_sha256"):
        raise ValueError("input manifest must pin nnunet_preprocessing_sha256")
    receipt = json.loads(args.preprocessing_evidence.read_text())
    raw_hashes, pre_hashes = _tree_hashes(raw), _tree_hashes(pre)
    if (receipt.get("schema_version") != "nnunet_preprocessing_v2" or receipt.get("dataset") != raw.name
            or receipt.get("raw_files") != raw_hashes or receipt.get("preprocessed_files") != pre_hashes):
        raise ValueError("raw/preprocessed files differ from preprocessing receipt")
    if receipt.get("runtime") != provenance.runtime_identity():
        raise ValueError("runtime differs from preprocessing receipt")
    if receipt.get("upstream") != _upstream_identity():
        raise ValueError("upstream source differs from preprocessing receipt")
    declared = inputs["manifest"]["raw_files"]
    if any(declared.get(str((raw / name).resolve())) != sha for name, sha in raw_hashes.items()):
        raise ValueError("input manifest must cover the actual nnUNet raw export")
    data = json.loads((raw / "dataset.json").read_text())
    if data != json.loads((pre / "dataset.json").read_text()):
        raise ValueError("raw/preprocessed dataset.json mismatch")
    if _positive_label_from_dataset_json(raw / "dataset.json") != _POSITIVE_LABEL[args.task]:
        raise ValueError("task/export positive label mismatch")
    if args.task.startswith("riga_") and data.get("ground_truth_identity") != provenance.ground_truth_identity(args.task):
        raise ValueError("historical RIGA nnUNet export rejected")
    splits = json.loads((pre / "splits_final.json").read_text())
    if splits != json.loads((raw / "splits_final.json").read_text()):
        raise ValueError("raw/preprocessed fold split mismatch")
    if not isinstance(splits, list) or not 0 <= args.fold < len(splits):
        raise ValueError("requested fold is absent; no random split fallback allowed")
    all_ids = set(_label_files(raw / "labelsTr"))
    for fold in splits:
        tr = _unique_ids(fold["train"], "fold training IDs")
        val = _unique_ids(fold["val"], "fold validation IDs")
        if set(tr) & set(val):
            raise ValueError("fold train/validation overlap")
        _same_ids(all_ids, tr + val, "GT universe")
        if args.task == "acdc_lv" and {cid.split("_frame")[0] for cid in tr} & {cid.split("_frame")[0] for cid in val}:
            raise ValueError("fold leaks ACDC patients")
    if data["numTraining"] != len(all_ids):
        raise ValueError("numTraining differs from GT universe")
    ending = data["file_ending"]
    channels = data["channel_names"]
    _same_ids([p.name for p in (raw / "imagesTr").iterdir()],
              [f"{cid}_{int(ch):04d}{ending}" for cid in all_ids for ch in channels], "training image IDs")
    plans = json.loads((pre / "nnUNetPlans.json").read_text())
    config = plans["configurations"][args.config]
    if (plans.get("dataset_name") != raw.name or plans.get("plans_name") != "nnUNetPlans"
            or "inherits_from" in config or config.get("previous_stage") or config.get("next_stage")):
        raise ValueError("dataset/plans mismatch or unsupported inherited/cascade configuration")
    identifier = config["data_identifier"]
    _unique_ids([identifier], "data_identifier")
    folder = pre / identifier
    numpy_ids = {p.stem for p in folder.glob("*.npz")}
    blosc_ids = {p.name[:-5] for p in folder.glob("*.b2nd") if not p.name.endswith("_seg.b2nd")}
    if numpy_ids and blosc_ids:
        raise ValueError("mixed preprocessed dataset formats")
    _same_ids(numpy_ids or blosc_ids, all_ids, "preprocessed case IDs")
    for cid in all_ids:
        if not (folder / f"{cid}.pkl").is_file() or (blosc_ids and not (folder / f"{cid}_seg.b2nd").is_file()):
            raise ValueError(f"missing preprocessed case components: {cid}")
    holdout_export = args.task.startswith("riga_") or (raw / "fmpool_test_ids.json").is_file()
    if args.evaluation == "holdout" and not (raw / "fmpool_test_ids.json").is_file():
        raise ValueError("holdout evaluation requires explicit holdout IDs in fmpool_test_ids.json")
    if args.task == "acdc_lv" and not holdout_export:
        # FM split IDs are slices; nnU-Net consumes complete frame volumes.
        declared_folds = inputs["split"].get("folds", {})
        for index, fold in enumerate(splits):
            for role in ("train", "val"):
                declared_ids = declared_folds.get(f"fold_{index}_{role}", [])
                if not declared_ids:
                    raise ValueError("input split must declare every ACDC fold")
                _same_ids(fold[role], {cid.split("_slice")[0] for cid in declared_ids}, "declared ACDC fold")
    holdout_ids = None
    if holdout_export:
        # Case-identical layout: inner CV over the declared training pool, one shared holdout.
        holdout_ids = _unique_ids(json.loads((raw / "fmpool_test_ids.json").read_text())["test_ids"], "holdout IDs")
        safe = lambda cid: cid.replace("/", "__").replace(" ", "_")
        _same_ids(all_ids, [safe(cid) for cid in inputs["split"]["train"]], "declared training IDs")
        _same_ids(holdout_ids, [safe(cid) for cid in inputs["split"]["test"]], "declared holdout IDs")
        if all_ids & set(holdout_ids):
            raise ValueError("holdout overlaps training pool")
        _same_ids(_label_files(raw / "labelsTs"), holdout_ids, "holdout GT IDs")
        declared_folds = inputs["split"].get("folds") or {}
        if args.task == "acdc_lv" and declared_folds and set(
                declared_folds.get("fold_0_train", []) + declared_folds.get("fold_0_val", [])) == all_ids:
            # Fixed-holdout design with declared inner folds: the export must use exactly those folds.
            if len(splits) != len(declared_folds) // 2:
                raise ValueError("declared ACDC fold count differs from the export")
            for index, fold in enumerate(splits):
                for role in ("train", "val"):
                    _same_ids(fold[role], declared_folds.get(f"fold_{index}_{role}", []), "declared ACDC fold")
        _same_ids([p.name for p in (raw / "imagesTs").iterdir()],
                  [f"{cid}_{int(ch):04d}{ending}" for cid in holdout_ids for ch in channels], "holdout image IDs")
    if ending == ".nii.gz":
        for role in ("Tr", "Ts") if holdout_ids else ("Tr",):
            for cid, label in _label_files(raw / f"labels{role}").items():
                for channel in channels:
                    image = raw / f"images{role}" / f"{cid}_{int(channel):04d}{ending}"
                    _validate_nifti_pair(image, label, cid)
    return {"raw": str(raw), "preprocessed": str(pre), "dataset": raw.name,
            "raw_files": raw_hashes, "preprocessed_files": pre_hashes,
            "fold": splits[args.fold], "holdout_ids": holdout_ids, "dataset_json": data,
            "plans": plans, "upstream": receipt["upstream"], "preprocessing_receipt_sha256": receipt_sha}


def main(argv=None):
    sys.dont_write_bytecode = True
    args = _parse_args(argv)
    output = args.out_json or args.out / f"seed_{args.seed}.json"
    cell = output.with_suffix(".cell")
    if output.exists() or cell.exists():
        raise FileExistsError(f"preserving existing output/cell: {output}")
    try:
        return _run(args, output, cell)
    except Exception as exc:
        _failure(output, exc)
        raise


def _run(args, output, cell):
    _set_nnunet_env(args)
    inputs = provenance.read_input_manifest(
        args.input_manifest, args.task, split_contract="nnunet_export")
    evidence = validate_workspace(args, inputs)
    cell.mkdir(parents=True, exist_ok=False)
    for key in ("raw", "preprocessed"):
        target = cell / key / evidence["dataset"]
        shutil.copytree(evidence[key], target)
        if _tree_hashes(target) != evidence[f"{key}_files"]:
            raise ValueError(f"{key} changed during private staging")
        os.environ[f"nnUNet_{key}"] = str(target.parent.absolute())
    os.environ["nnUNet_results"] = str((cell / "results").absolute())
    # Upstream caches paths at import time. Never train with stale module globals.
    if "nnunetv2.paths" in sys.modules:
        raise RuntimeError("nnunetv2.paths already imported; launch a fresh process")
    from fmpool import nnunet_seeded
    nnunet_seeded.ensure_nnunet_discovery(write=False)
    from nnunetv2.run.run_training import run_training
    from nnunetv2.utilities.file_path_utilities import get_output_folder
    determinism.set_seed(args.seed)
    trainer = nnunet_seeded.released_alias_name(args.base_trainer, args.seed)
    source = {name: (Path(__file__).parent / name).read_text() for name in
              ("train_nnunet_baseline.py", "predict_nnunet_holdout.py", "run_nnunet.sh", "run_nnunet_case_worker.sh")}
    identity = provenance.run_identity(args, __file__, inputs=inputs, data_identity=evidence,
        recipe={"version": "nnunet_external_v2", "precision": "upstream_trainer_managed"})
    prepared = {"schema_version": "nnunet_cell_v2", "run_identity_v2": identity,
                "entrypoint_sources": source, "data": evidence, "task": args.task,
                "dataset_id": args.dataset_id, "config": args.config, "fold": args.fold,
                "seed": args.seed, "base_trainer": args.base_trainer, "trainer": trainer,
                "evaluation": args.evaluation, "environment": {key: os.environ[key] for key in
                    ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")}}
    _write_json(cell / "prepared.json", prepared)
    started = time.time()
    run_training(dataset_name_or_id=evidence["dataset"], configuration=args.config, fold=args.fold,
                 trainer_class_name=trainer, plans_identifier="nnUNetPlans", continue_training=False)
    model = Path(get_output_folder(evidence["dataset"], trainer, "nnUNetPlans", args.config))
    checkpoint = model / f"fold_{args.fold}" / "checkpoint_final.pth"
    completed = {**prepared, "status": "trained", "model": str(model.absolute()),
                 "checkpoint_sha256": determinism.sha256_file(checkpoint),
                 "training_elapsed_s": time.time() - started}
    # Record any upstream-created unpacked data too; original pinned inputs must remain unchanged.
    private_pre = Path(os.environ["nnUNet_preprocessed"]) / evidence["dataset"]
    after = _tree_hashes(private_pre)
    if any(after.get(name) != sha for name, sha in evidence["preprocessed_files"].items()):
        raise ValueError("pinned preprocessing changed during training")
    completed["consumed_preprocessed_files"] = after
    _write_json(cell / "trained.json", completed)
    from predict_nnunet_holdout import score_training
    score_training(cell / "trained.json", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
