#!/usr/bin/env python3
"""Materialise RIGA Cup + ACDC LV into nnU-Net v2 raw layout.

Usage
-----
    # In the raw-data environment
    export nnUNet_raw=nnunet_workspace/raw
    export nnUNet_preprocessed=nnunet_workspace/preprocessed
    export nnUNet_results=nnunet_workspace/results
    python3 code/scripts/prepare_nnunet_data.py --task riga_cup --dataset-id 501
    python3 code/scripts/prepare_nnunet_data.py --task acdc_lv  --dataset-id 502

Layout produced
---------------
    $nnUNet_raw/Dataset501_RIGACup/
        imagesTr/<case_id>_{0000,0001,0002}.png (RIGA Cup train RGB channels)
        labelsTr/<case_id>.png                 (train binary 0/1, 4-of-6 majority)
        imagesTs/<case_id>_{0000,0001,0002}.png (Magrabia held-out RGB channels)
        labelsTs/<case_id>.png                 (held-out labels for scoring only)
        dataset.json
        splits_final.json                      (5-fold over train pool)
        fmpool_test_ids.json                   (Magrabia held-out)

    $nnUNet_raw/Dataset502_ACDCLV/
        imagesTr/<patient>_frame{ED,ES}_0000.nii.gz  (primary-train patients: 3D volume)
        labelsTr/<patient>_frame{ED,ES}.nii.gz       (binary {0,1}, LV only)
        imagesTs/, labelsTs/                   (primary-test patients, scoring only)
        dataset.json
        splits_final.json                      (5-fold patient-level over train pool)
        fmpool_test_ids.json                   (primary-test frames, case-identical)

The case_ids written here align with the loader-managed split JSONs under
``data/splits/`` so per-case Dice keys match between the FM models and the
nnU-Net baselines.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Ensure local src/ wins over any installed copy.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from fmpool.datasets import load_committed_split, resolve_task_root  # noqa: E402

logger = logging.getLogger("fmpool.prepare_nnunet_data")


# ---------------------------------------------------------------------------
# Dataset name conventions
# ---------------------------------------------------------------------------

_TASK_TO_DS_NAME = {
    "riga_cup": "RIGACup",
    "acdc_lv": "ACDCLV",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=sorted(_TASK_TO_DS_NAME))
    p.add_argument("--dataset-id", type=int, required=True)
    p.add_argument(
        "--nnunet-raw",
        default=os.environ.get("nnUNet_raw"),
        help="Override nnUNet_raw root; defaults to env var.",
    )
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--preprocess", action="store_true", help="Preprocess an existing export into a fresh owned workspace")
    p.add_argument("--preprocessing-evidence", type=Path)
    p.add_argument("--nnunet-preprocessed", default=os.environ.get("nnUNet_preprocessed"))
    p.add_argument("--split-json", type=Path, help="ACDC only: frame-level nnU-Net split (case_unit=frame_volume, "
                   "declared folds) instead of deriving one from the committed slice split")
    return p.parse_args(argv)


def _raw_dir(raw_root: Path, dataset_id: int, name: str) -> Path:
    return raw_root / f"Dataset{dataset_id:03d}_{name}"


def _safe_case_id(cid: str) -> str:
    """nnU-Net uses '_' as channel delimiter; '/' is illegal in filenames."""
    return cid.replace("/", "__").replace(" ", "_")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# RIGA Cup (2D)
# ---------------------------------------------------------------------------


def _export_riga_cup(raw_root: Path, dataset_id: int) -> Path:
    """RIGA Cup: 2D fundus PNG + 4-of-6 majority cup binary mask."""
    from PIL import Image

    from fmpool.datasets.riga import RIGADataset, GROUND_TRUTH_IDENTITY

    out_dir = _raw_dir(raw_root, dataset_id, _TASK_TO_DS_NAME["riga_cup"])
    images_dir = out_dir / "imagesTr"
    labels_dir = out_dir / "labelsTr"
    test_images_dir = out_dir / "imagesTs"
    test_labels_dir = out_dir / "labelsTs"
    ds_json = out_dir / "dataset.json"
    if out_dir.exists():
        raise FileExistsError("Preserve existing RIGA export; choose a new dataset ID")
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)

    train_ds = RIGADataset(target="cup", split="train")
    test_ds = RIGADataset(target="cup", split="test")

    written: list[str] = []
    nnunet_train_ids: list[str] = []
    nnunet_test_ids: list[str] = []

    def _dump(
        ds,
        bucket: list[str],
        image_out_dir: Path,
        label_out_dir: Path,
    ) -> None:
        for image_t, mask_t, case_id in ds:
            safe = _safe_case_id(case_id)
            img_np = image_t.permute(1, 2, 0).numpy()  # HxWx3 uint8
            mask_np = mask_t.squeeze(0).numpy().astype(np.uint8)  # 0/1
            for ch in range(3):
                Image.fromarray(img_np[:, :, ch], mode="L").save(
                    image_out_dir / f"{safe}_{ch:04d}.png"
                )
            Image.fromarray(mask_np, mode="L").save(
                label_out_dir / f"{safe}.png"
            )
            bucket.append(safe)
            written.append(safe)

    _dump(train_ds, nnunet_train_ids, images_dir, labels_dir)
    _dump(test_ds, nnunet_test_ids, test_images_dir, test_labels_dir)

    # 5-fold over the *training pool* (BinRushed + MESSIDOR). The held-out
    # Magrabia pool (test set) is exported under imagesTs/labelsTs, so
    # nnU-Net planning/preprocessing sees only training cases.
    rng = np.random.default_rng(42)
    perm = list(nnunet_train_ids)
    rng.shuffle(perm)
    n = len(perm)
    folds: list[dict] = []
    for k in range(5):
        lo = (n * k) // 5
        hi = (n * (k + 1)) // 5
        val = perm[lo:hi]
        train = [c for c in perm if c not in set(val)]
        folds.append({"train": train, "val": val})
    _write_json(out_dir / "splits_final.json", folds)
    _write_json(
        out_dir / "fmpool_test_ids.json",
        {"test_ids": nnunet_test_ids},
    )

    dataset_json = {
        "channel_names": {
            "0": "fundus_R",
            "1": "fundus_G",
            "2": "fundus_B",
        },
        "labels": {"background": 0, "cup": 1},
        "numTraining": len(nnunet_train_ids),
        "file_ending": ".png",
        "name": "RIGACup",
        "ground_truth_identity": GROUND_TRUTH_IDENTITY + ":riga_cup",
        "training_palette_audit": train_ds.palette_audit,
        "test_palette_audit": test_ds.palette_audit,
        "description": "RIGA+ optic cup, 4-of-6 rater majority.",
    }
    _write_json(ds_json, dataset_json)
    logger.info(
        "RIGA Cup: wrote %d cases (train=%d test=%d) under %s",
        len(written),
        len(nnunet_train_ids),
        len(nnunet_test_ids),
        out_dir,
    )
    return out_dir


# ---------------------------------------------------------------------------
# ACDC LV (3D)
# ---------------------------------------------------------------------------


def _export_acdc_lv(raw_root: Path, dataset_id: int, split: dict | None = None) -> Path:
    """ACDC LV: 3D volumes, label binarised to {0, 1=LV}.

    Case-identical layout: mirrors RIGA. The split's train
    patients form the nnU-Net pool (5-fold patient CV); its test patients go to
    imagesTs/labelsTs so that every fold/seed is scored on the same held-out
    frames. ``split`` (``--split-json``) is a frame-level split with declared
    folds (official 100/50 patient split); without it the committed slice split
    (``acdc_official_split.json``, the same official patients) is used and seed-42
    folds are generated. Training folds do not include the held-out test patients.
    """
    import nibabel as nib
    import SimpleITK as sitk

    from fmpool.datasets.acdc import _FRAME_RE, _LV_LABEL, acdc_mask_path  # type: ignore
    from train_nnunet_baseline import _validate_nifti_pair

    out_dir = _raw_dir(raw_root, dataset_id, _TASK_TO_DS_NAME["acdc_lv"])
    images_dir = out_dir / "imagesTr"
    labels_dir = out_dir / "labelsTr"
    ds_json = out_dir / "dataset.json"
    if out_dir.exists():
        raise FileExistsError("Preserve existing ACDC export; choose a new dataset ID")
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    root = resolve_task_root("acdc")
    images_src = root / "Images"
    masks_src = root / "Masks"

    if split is None:
        split = load_committed_split("acdc_lv")
    declared_folds = split.get("folds") if split.get("case_unit") == "frame_volume" else None
    if declared_folds is not None:
        from fmpool.provenance import validate_source_split
        validate_source_split("acdc_lv", split, split_contract="nnunet_export")
    frames = lambda ids: set(cid.split("_slice")[0] for cid in ids)  # noqa: E731
    train_frames, test_frames = frames(split["train"]), frames(split["test"])
    if not train_frames or not test_frames or (
            {f.split("_frame")[0] for f in train_frames} & {f.split("_frame")[0] for f in test_frames}):
        raise ValueError("committed ACDC split must give patient-disjoint train/test frames")

    frame_paths: list[tuple[str, str, Path, Path]] = []
    for entry in sorted(images_src.iterdir()):
        m = _FRAME_RE.match(entry.name)
        if m is None:
            continue
        patient = m.group(1)
        frame = m.group(2)
        mask = acdc_mask_path(masks_src, entry.name)  # plain-name masks only with FMPOOL_ACDC_ALLOW_PLAIN_MASKS=1
        frame_paths.append((patient, frame, entry, mask))
    on_disk = {f"{p}_frame{f}" for p, f, _, _ in frame_paths}
    if not (train_frames | test_frames) <= on_disk:
        raise FileNotFoundError(f"split frames missing on disk: {sorted((train_frames | test_frames) - on_disk)[:5]}")
    test_images_dir = out_dir / "imagesTs"
    test_labels_dir = out_dir / "labelsTs"
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    test_ids: list[str] = []
    patient_to_cases: dict[str, list[str]] = {}
    for patient, frame, img_path, mask_path in frame_paths:
        safe = _safe_case_id(f"{patient}_frame{frame}")
        if safe not in train_frames and safe not in test_frames:
            continue
        is_test = safe in test_frames
        img = nib.load(str(img_path))
        msk = nib.load(str(mask_path))
        img_data = np.asarray(img.dataobj)
        msk_data = np.asarray(msk.dataobj)
        if not np.isfinite(img_data).all() or not np.isfinite(msk_data).all():
            raise ValueError(f"nonfinite ACDC image/mask: {safe}")
        image_grid, mask_grid = _validate_nifti_pair(img_path, mask_path, safe)
        msk_bin = (msk_data == _LV_LABEL).astype(np.uint8)
        binary = sitk.GetImageFromArray(msk_bin.transpose(2, 1, 0))
        binary.CopyInformation(mask_grid)
        # Canonicalize each file from its own validated reader geometry; never resample.
        sitk.WriteImage(sitk.Cast(image_grid, sitk.sitkFloat32),
                        str((test_images_dir if is_test else images_dir) / f"{safe}_0000.nii.gz"))
        sitk.WriteImage(binary, str((test_labels_dir if is_test else labels_dir) / f"{safe}.nii.gz"))
        if is_test:
            test_ids.append(safe)
            continue
        written.append(safe)
        patient_to_cases.setdefault(patient, []).append(safe)

    patients = sorted(patient_to_cases)
    folds: list[dict] = []
    if declared_folds is not None:
        for k in range(len(declared_folds) // 2):
            folds.append({"train": sorted(declared_folds[f"fold_{k}_train"]), "val": sorted(declared_folds[f"fold_{k}_val"])})
    else:
        # Patient-level 5-fold CV over the training pool (seed=42 shuffle).
        rng = np.random.default_rng(42)
        shuffled = list(patients)
        rng.shuffle(shuffled)
        n = len(shuffled)
        for k in range(5):
            val_pat = set(shuffled[(n * k) // 5:(n * (k + 1)) // 5])
            val_cases: list[str] = []
            train_cases: list[str] = []
            for p, cases in patient_to_cases.items():
                (val_cases if p in val_pat else train_cases).extend(cases)
            folds.append({"train": sorted(train_cases), "val": sorted(val_cases)})
    _write_json(out_dir / "splits_final.json", folds)
    _write_json(out_dir / "fmpool_test_ids.json", {"test_ids": sorted(test_ids)})

    dataset_json = {
        "channel_names": {"0": "cine"},
        "labels": {"background": 0, "LV": 1},
        "numTraining": len(written),
        "file_ending": ".nii.gz",
        "name": "ACDCLV",
        "ground_truth_identity": "acdc_lv_gt_v1:export_binary1",
        "description": "ACDC LV binary (paper binary task).",
    }
    _write_json(ds_json, dataset_json)
    logger.info(
        "ACDC LV: wrote %d volumes from %d patients under %s",
        len(written),
        len(patients),
        out_dir,
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    if not args.nnunet_raw:
        print(
            "[prepare_nnunet_data] ERROR: --nnunet-raw or env nnUNet_raw required",
            file=sys.stderr,
        )
        return 2
    raw_root = Path(args.nnunet_raw)
    if args.preprocess:
        if not args.preprocessing_evidence or not args.nnunet_preprocessed:
            raise ValueError("preprocessing requires --preprocessing-evidence and nnUNet_preprocessed")
        from train_nnunet_baseline import prepare_preprocessing
        prepare_preprocessing(_raw_dir(raw_root, args.dataset_id, _TASK_TO_DS_NAME[args.task]),
                              Path(args.nnunet_preprocessed), args.preprocessing_evidence)
        return 0
    raw_root.mkdir(parents=True, exist_ok=True)

    if args.task == "riga_cup":
        _export_riga_cup(raw_root, args.dataset_id)
    elif args.task == "acdc_lv":
        split = json.loads(args.split_json.read_text()) if args.split_json else None
        _export_acdc_lv(raw_root, args.dataset_id, split=split)
    else:
        raise ValueError(args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
