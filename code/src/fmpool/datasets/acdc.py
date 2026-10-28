"""ACDC cardiac MRI left-ventricle loader (task ``acdc_lv``).

Each patient has two cine frames (ED, ES). We extract *all axial slices
with non-empty LV label* per frame and treat each slice as a case.

The paper uses the official challenge split (default, ``FMPOOL_ACDC_SPLIT=official``): the committed
``acdc_official_split.json`` trains on all LV-positive ED/ES slices of the 100 official training
patients (001-100) and tests on those of the 50 official testing patients (101-150), whose references
are published on the ACDC site (never generated here; :func:`validate_acdc_official_split`).
``FMPOOL_ACDC_SPLIT=cv5`` selects a superseded patient-level 5-fold CV over patients 001-100
(``numpy`` seed 42, fold 0), which is generated and saved on first use.

GT masks are read from the official ``Masks/patientXXX_frameYY_gt.nii.gz`` names only. A
mask stored under the plain image name (``patientXXX_frameYY.nii.gz``) is refused with a
``FileNotFoundError`` unless the caller opts in with ``FMPOOL_ACDC_ALLOW_PLAIN_MASKS=1``
(:func:`acdc_mask_path`); plain-name masks require explicit opt-in.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from fmpool.datasets import (
    acdc_split_name,
    load_committed_split,
    resize_mask,
    require_dir,
    resolve_task_root,
    save_committed_split,
)

_NUM_FOLDS = 5
_PATIENTS_PER_FOLD = 20
_PUBLIC_PATIENT_MAX = 100
_TEST_PATIENT_RANGE = range(101, 151)  # official ACDC testing set
OFFICIAL_SCHEME = "official_bernard2018_100_50"
_CASE_RE = re.compile(r"(patient\d{3})_frame(\d{2})_slice(\d{3,})")
_LV_LABEL = 3
_TARGET_HW = 224
_FRAME_RE = re.compile(r"^(patient\d{3})_frame(\d{2})\.nii\.gz$")
_PLAIN_MASK_ENV = "FMPOOL_ACDC_ALLOW_PLAIN_MASKS"


def acdc_mask_path(masks_dir: Path, image_name: str) -> Path:
    """GT mask for ``image_name``: ``*_gt.nii.gz``; the plain name only with the opt-in env."""
    gt = masks_dir / image_name.replace(".nii.gz", "_gt.nii.gz")
    if gt.is_file():
        return gt
    plain = masks_dir / image_name
    if not plain.is_file():
        raise FileNotFoundError(
            f"[acdc] mask missing for {image_name}: expected {gt} "
            f"(or {plain} with {_PLAIN_MASK_ENV}=1)"
        )
    if os.environ.get(_PLAIN_MASK_ENV, "").strip() != "1":
        raise FileNotFoundError(
            f"[acdc] {gt} missing; refusing plain-named {plain}. "
            f"Set {_PLAIN_MASK_ENV}=1 to opt in to plain-named masks."
        )
    return plain


def validate_acdc_split(manifest: dict, case_index: dict | None = None) -> None:
    """Bind all folds and primary roles to canonical raw-derived slice IDs."""
    cases = manifest["_all_cases"]
    if not cases or len(cases) != len(set(cases)):
        raise ValueError("ACDC split has empty or duplicate cases")
    canonical = {}
    for cid in cases:
        match = re.fullmatch(r"(patient\d{3})_frame(\d{2})_slice(\d{3,})", cid)
        if match is None or not 1 <= int(match[1][-3:]) <= _PUBLIC_PATIENT_MAX:
            raise ValueError(f"ACDC noncanonical public case: {cid}")
        canonical[cid] = (match[1], match[2], int(match[3]))
    if case_index is not None and canonical != case_index:
        raise ValueError("ACDC split differs from the actual raw case index")
    folds, validation_cases = manifest["folds"], set()
    expected_keys = {f"fold_{k}_{role}" for k in range(_NUM_FOLDS) for role in ("train", "val")}
    if set(folds) != expected_keys:
        raise ValueError("ACDC requires all five train/validation folds")
    for k in range(_NUM_FOLDS):
        train, val = folds[f"fold_{k}_train"], folds[f"fold_{k}_val"]
        if (len(train + val) != len(set(train + val)) or set(train + val) != set(cases)
                or validation_cases & set(val)):
            raise ValueError(f"ACDC fold {k} has missing, duplicate or overlapping cases")
        train_patients = {canonical[cid][0] for cid in train}
        val_patients = {canonical[cid][0] for cid in val}
        if train_patients & val_patients or len(val_patients) != _PATIENTS_PER_FOLD:
            raise ValueError(f"ACDC fold {k} is not a disjoint twenty-patient validation fold")
        validation_cases.update(val)
    if validation_cases != set(cases):
        raise ValueError("ACDC validation folds do not cover the raw cohort")
    if (manifest.get("train") != folds["fold_0_train"] or manifest.get("val") != []
            or manifest.get("test") != folds["fold_0_val"]):
        raise ValueError("ACDC primary roles differ from fold zero")


def validate_acdc_official_frames(frames: set[str]) -> None:
    """Bind ED/ES identities to the committed frame listing, independently of regenerated slices."""
    by_patient: dict[str, set[str]] = {}
    for frame in frames:
        match = _FRAME_RE.fullmatch(f"{frame}.nii.gz")
        if match is None:
            raise ValueError(f"ACDC noncanonical official frame: {frame}")
        by_patient.setdefault(match[1], set()).add(match[2])
    if (set(by_patient) != {f"patient{p:03d}" for p in range(1, _TEST_PATIENT_RANGE.stop)}
            or any(len(phases) != 2 for phases in by_patient.values())):
        raise ValueError("ACDC official cohort requires exactly two distinct ED/ES frames for each patient001-150")
    committed = load_committed_split("acdc_lv_nnunet_official")
    if committed is None:
        raise FileNotFoundError("acdc_lv_nnunet_official_split.json is required to validate official ED/ES frames")
    expected = set(committed["train"] + committed["test"])
    if frames != expected:
        raise ValueError(f"ACDC official ED/ES frames differ from the committed frame listing: "
                         f"missing {sorted(expected - frames)[:5]}, unexpected {sorted(frames - expected)[:5]}")


def validate_acdc_official_split(manifest: dict, case_index: dict | None = None) -> None:
    """Official split: train = patients 001-100 (all), test = 101-150 (all); slice IDs bound to the raw index."""
    if manifest.get("split_scheme") != OFFICIAL_SCHEME or manifest.get("val") != []:
        raise ValueError("ACDC official split requires split_scheme official_bernard2018_100_50 and an empty val")
    cases = manifest["_all_cases"]
    train, test = manifest["train"], manifest["test"]
    if not cases or len(cases) != len(set(cases)) or sorted(train + test) != sorted(cases) or set(train) & set(test):
        raise ValueError("ACDC official split roles do not partition _all_cases")
    canonical = {}
    for cid in cases:
        match = _CASE_RE.fullmatch(cid)
        if match is None or not 1 <= int(match[1][-3:]) <= _TEST_PATIENT_RANGE.stop - 1:
            raise ValueError(f"ACDC noncanonical case: {cid}")
        canonical[cid] = (match[1], match[2], int(match[3]))
    if case_index is not None and canonical != case_index:
        raise ValueError("ACDC official split differs from the actual raw case index")
    train_patients = {canonical[c][0] for c in train}
    test_patients = {canonical[c][0] for c in test}
    if train_patients != {f"patient{p:03d}" for p in range(1, _PUBLIC_PATIENT_MAX + 1)}:
        raise ValueError("ACDC official train must cover exactly patients 001-100")
    if test_patients != {f"patient{p:03d}" for p in _TEST_PATIENT_RANGE}:
        raise ValueError("ACDC official test must cover exactly patients 101-150")
    validate_acdc_official_frames({f"{patient}_frame{frame}" for patient, frame, _ in canonical.values()})


def validate_any_acdc_split(manifest: dict, case_index: dict | None = None) -> None:
    """Dispatch on the split scheme (official vs the superseded 5-fold CV manifest)."""
    if manifest.get("split_scheme") == OFFICIAL_SCHEME:
        validate_acdc_official_split(manifest, case_index)
    else:
        validate_acdc_split(manifest, case_index)


class ACDCLVDataset(Dataset):
    """ACDC LV binary segmentation (task ``acdc_lv``).

    Returns ``(image_uint8 [3,224,224], mask_bool [1,224,224], case_id)``.
    ``case_id`` format: ``"{patient}_frame{frame}_slice{slice_idx:03d}"``.
    Split values accepted: ``fold_{k}_train`` / ``fold_{k}_val`` for
    ``k ∈ {0..4}`` (CV manifest only), plus plain ``train`` / ``test``: the official
    100/50-patient partition by default, or fold 0 under ``FMPOOL_ACDC_SPLIT=cv5``.
    """

    def __init__(
        self,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.split = split
        self.transform = transform
        self.root = resolve_task_root("acdc")
        require_dir(self.root, dataset="acdc")
        self.images_dir = self.root / "Images"
        self.masks_dir = self.root / "Masks"
        require_dir(self.images_dir, dataset="acdc")
        require_dir(self.masks_dir, dataset="acdc")

        public_patients, all_patients = self._discover_patients()
        if len(public_patients) != _PUBLIC_PATIENT_MAX:
            raise RuntimeError(
                f"[acdc] expected {_PUBLIC_PATIENT_MAX} public patients "
                f"(001-100), found {len(public_patients)}. All patients "
                f"on disk: {len(all_patients)}"
            )
        self.official = acdc_split_name() == "acdc_official"
        if self.official:
            test_patients = [p for p in all_patients if int(p[-3:]) in _TEST_PATIENT_RANGE]
            if len(test_patients) != len(_TEST_PATIENT_RANGE):
                raise RuntimeError(f"[acdc] official split needs patients 101-150 on disk, found {len(test_patients)}")
            manifest = self._load_official_split(public_patients + test_patients)
        else:
            manifest = self._load_or_generate_split(public_patients)
        self.case_ids: list[str] = self._resolve_split(split, manifest)

    def _discover_patients(self) -> tuple[list[str], list[str]]:
        """Return (public patients [001-100], all patients on disk)."""
        all_patients: set[str] = set()
        for entry in self.images_dir.iterdir():
            m = _FRAME_RE.match(entry.name)
            if m is None:
                continue
            all_patients.add(m.group(1))
        all_sorted = sorted(all_patients)
        public = [p for p in all_sorted if 1 <= int(p[-3:]) <= _PUBLIC_PATIENT_MAX]
        return public, all_sorted

    def _frame_paths(
        self, patient: str
    ) -> list[tuple[str, Path, Path]]:
        frames: list[tuple[str, Path, Path]] = []
        for entry in sorted(self.images_dir.iterdir()):
            m = _FRAME_RE.match(entry.name)
            if m is None or m.group(1) != patient:
                continue
            frame_label = m.group(2)
            mask = acdc_mask_path(self.masks_dir, entry.name)
            frames.append((frame_label, entry, mask))
        if not frames:
            raise RuntimeError(f"[acdc] no frames found for {patient}")
        return frames

    def _enumerate_cases(
        self, patients: list[str], *, official: bool = False
    ) -> tuple[list[str], dict[str, tuple[str, str, int]]]:
        import nibabel as nib  # local: heavy optional dep

        frames = {patient: self._frame_paths(patient) for patient in patients}
        if official:
            validate_acdc_official_frames({f"{patient}_frame{frame}" for patient in patients
                                           for frame, _, _ in frames[patient]})
        case_ids: list[str] = []
        index: dict[str, tuple[str, str, int]] = {}
        for patient in patients:
            for frame, _img_path, mask_path in frames[patient]:
                mask_arr = nib.load(str(mask_path)).get_fdata()
                for slice_idx in range(mask_arr.shape[2]):
                    slc = mask_arr[:, :, slice_idx]
                    if not np.any(slc == _LV_LABEL):
                        continue
                    case_id = (
                        f"{patient}_frame{frame}_slice{slice_idx:03d}"
                    )
                    case_ids.append(case_id)
                    index[case_id] = (patient, frame, slice_idx)
        self._case_index = index
        return case_ids, index

    def _load_official_split(self, patients: list[str]) -> dict:
        """Committed official split only (never generated): validated against the raw slice index of all 150 patients."""
        committed = load_committed_split("acdc_lv")  # -> acdc_official_split.json (default split)
        if committed is None:
            raise FileNotFoundError("acdc_official_split.json is missing")
        _, self._case_index = self._enumerate_cases(patients, official=True)
        validate_acdc_official_split(committed, self._case_index)
        return {"_all_cases": list(committed["_all_cases"]), "folds": {},
                "train": list(committed["train"]), "test": list(committed["test"])}

    def _load_or_generate_split(
        self, public_patients: list[str]
    ) -> dict:
        committed = load_committed_split("acdc_lv")
        if committed is not None:
            _, self._case_index = self._enumerate_cases(public_patients)
            validate_acdc_split(committed, self._case_index)
            return {
                "_all_cases": list(committed.get("_all_cases", [])),
                "folds": committed.get("folds", {}),
            }

        case_ids, _ = self._enumerate_cases(public_patients)

        rng = np.random.default_rng(42)
        shuffled = list(public_patients)
        rng.shuffle(shuffled)
        folds: dict[str, list[str]] = {}
        for k in range(_NUM_FOLDS):
            lo = k * _PATIENTS_PER_FOLD
            hi = lo + _PATIENTS_PER_FOLD
            val_patients = set(shuffled[lo:hi])
            train_patients = set(shuffled) - val_patients
            folds[f"fold_{k}_val"] = [
                cid
                for cid in case_ids
                if self._case_index[cid][0] in val_patients
            ]
            folds[f"fold_{k}_train"] = [
                cid
                for cid in case_ids
                if self._case_index[cid][0] in train_patients
            ]

        provenance = (
            "Patient-level 5-fold CV over ACDC patients 001-100 "
            ". numpy.random.default_rng(seed=42).shuffle on "
            "patient list, 20 patients/fold. Patients 101-150 (with GT "
            "on disk but excluded by this CV split) are excluded. Slice cases "
            "retained where label==3 (LV) has any positive pixel."
        )
        persisted = {
            "train": folds["fold_0_train"],
            "val": [],
            "test": folds["fold_0_val"],
            "folds": folds,
            "_all_cases": case_ids,
        }
        validate_acdc_split(persisted, self._case_index)
        save_committed_split("acdc_lv", persisted, provenance=provenance)
        return {"_all_cases": case_ids, "folds": folds}

    @staticmethod
    def _resolve_split(split: str, manifest: dict) -> list[str]:
        folds = manifest.get("folds", {})
        if split in folds:
            return list(folds[split])
        if split == "train":
            return list(manifest.get("train", folds.get("fold_0_train", [])))
        if split == "val":
            return []
        if split == "test":
            return list(manifest.get("test", folds.get("fold_0_val", [])))
        raise ValueError(
            f"ACDC split must be 'train', 'val', 'test', or "
            f"'fold_{{k}}_{{train,val}}'; got {split!r}"
        )

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        case_id = self.case_ids[idx]
        patient, frame, slice_idx = self._case_index[case_id]
        img_path = self.images_dir / f"{patient}_frame{frame}.nii.gz"
        mask_path = acdc_mask_path(self.masks_dir, img_path.name)

        import nibabel as nib  # local: heavy optional dep

        vol = nib.load(str(img_path)).get_fdata()
        seg = nib.load(str(mask_path)).get_fdata()
        img_slice = np.asarray(vol[:, :, slice_idx], dtype=np.float32)
        seg_slice = np.asarray(seg[:, :, slice_idx])

        lo, hi = float(img_slice.min()), float(img_slice.max())
        if hi > lo:
            img_u8 = (
                (img_slice - lo) / (hi - lo) * 255.0
            ).astype(np.uint8)
        else:
            img_u8 = np.zeros_like(img_slice, dtype=np.uint8)

        img_t = torch.from_numpy(img_u8).unsqueeze(0).repeat(3, 1, 1)
        mask_bool = seg_slice == _LV_LABEL
        mask_t = torch.from_numpy(mask_bool).unsqueeze(0)

        img_t = _resize_uint8(img_t, _TARGET_HW)
        mask_t = _resize_bool(mask_t, _TARGET_HW)

        if self.transform is not None:
            img_t, mask_t = self.transform(img_t, mask_t)
        return img_t, mask_t, case_id


def _resize_uint8(img: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinear resize a uint8 [C,H,W] tensor; return uint8 [C,size,size]."""
    import torch.nn.functional as F

    x = img.to(torch.float32).unsqueeze(0)
    x = F.interpolate(
        x, size=(size, size), mode="bilinear", align_corners=False
    )
    return x.clamp(0, 255).squeeze(0).to(torch.uint8)


def _resize_bool(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Nearest resize a bool [1,H,W] tensor; return bool [1,size,size]."""
    x = mask.to(torch.float32).unsqueeze(0)
    x = resize_mask(x, (size, size))  # nearest unless FMPOOL_MASK_RESIZE=nearest-exact
    return x.squeeze(0).to(torch.bool)
