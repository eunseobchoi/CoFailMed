"""BraTS 2024 GLI non-background foreground loader.

The binary target is ``seg > 0`` across BraTS 2024 GLI labels {0,1,2,3,4},
including resection cavities. The stable task identifier is ``brats_wt``;
this 2D foreground derivative is not the official 3D BraTS WT target.

Expected per-subject layout:
  <task_root>/training/training_data1_v2/<subject>/<subject>-{t1c,t1n,t2f,t2w,seg}.nii.gz
  <task_root>/additional/training_data_additional/<subject>/<subject>-{t1c,t1n,t2f,t2w,seg}.nii.gz

FLAIR-only 2D axial slice protocol. For each subject, keep
up to the top-10 axial slices by tumour area, minimum 64 positive
pixels. Patient-level 80/20 split via stdlib seed-42 over sorted patient
prefixes, retaining every session together.
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from fmpool.case_ids import brats_patient_split, validate_brats_split
from fmpool.datasets import (
    load_committed_split,
    resize_mask,
    require_dir,
    resolve_task_root,
    save_committed_split,
)

_MIN_TUMOUR_PIXELS = 64
_TOP_K_SLICES = 10
_TARGET_HW = 224
_SPLIT_SEED = 42


class BraTSWTDataset(Dataset):
    """BraTS 2D axial-slice non-background segmentation (``brats_wt`` id).

    Returns ``(image_uint8 [3,224,224], mask_bool [1,224,224], case_id)``
    with ``case_id = f"{subject_id}_slice{axial_idx:03d}"``.
    """

    def __init__(
        self,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"BraTS split must be train/val/test, got {split!r}"
            )
        self.split = split
        self.transform = transform
        self.root = resolve_task_root("brats")
        require_dir(self.root, dataset="brats")

        self.subject_dirs = self._discover_subjects()
        if not self.subject_dirs:
            raise RuntimeError(
                f"[brats] no subjects under "
                f"{self.root}/(training|additional). Expected unzipped "
                f"BraTS2024-BraTS-GLI training archives."
            )

        manifest = self._load_or_generate_split()
        self._slice_index: dict[str, tuple[str, int]] = {
            cid: (sub, axi)
            for cid, sub, axi in manifest["_case_entries"]
        }
        self.case_ids: list[str] = list(manifest[split])

    def _discover_subjects(self) -> dict[str, Path]:
        subjects: dict[str, Path] = {}
        candidates = [
            self.root / "training" / "training_data1_v2",
            self.root / "additional" / "training_data_additional",
        ]
        for parent in candidates:
            if not parent.is_dir():
                continue
            for sub_dir in sorted(parent.iterdir()):
                if not sub_dir.is_dir():
                    continue
                sub_id = sub_dir.name
                if sub_id.startswith("BraTS-GLI-"):
                    missing = [m for m in ('seg', 't2f')
                               if not (sub_dir / f"{sub_id}-{m}.nii.gz").is_file()]
                    if missing:
                        raise FileNotFoundError(f"incomplete BraTS session {sub_id}: {missing}")
                seg = sub_dir / f"{sub_id}-seg.nii.gz"
                flair = sub_dir / f"{sub_id}-t2f.nii.gz"
                if seg.is_file() and flair.is_file():
                    if sub_id in subjects:
                        raise ValueError(f"duplicate BraTS session directory: {sub_id}")
                    subjects[sub_id] = sub_dir
        return subjects

    @staticmethod
    def _pick_slices(seg_vol: np.ndarray) -> list[int]:
        areas = (seg_vol > 0).reshape(-1, seg_vol.shape[-1]).sum(axis=0)
        keep = [
            int(i) for i, a in enumerate(areas) if a >= _MIN_TUMOUR_PIXELS
        ]
        keep.sort(key=lambda i: -int(areas[i]))
        return sorted(keep[:_TOP_K_SLICES])

    def _build_case_entries(self) -> list[tuple[str, str, int]]:
        import nibabel as nib  # local: heavy optional dep

        entries: list[tuple[str, str, int]] = []
        for sub_id, sub_dir in self.subject_dirs.items():
            seg_path = sub_dir / f"{sub_id}-seg.nii.gz"
            seg_vol = nib.load(str(seg_path)).get_fdata()
            for axi in self._pick_slices(seg_vol):
                case_id = f"{sub_id}_slice{axi:03d}"
                entries.append((case_id, sub_id, axi))
        return entries

    def _load_or_generate_split(self) -> dict:
        # Check raw-derived entries even when a manifest is already committed.
        case_entries = self._build_case_entries()
        committed = load_committed_split("brats_wt")
        if committed is not None:
            validate_brats_split(committed, self.subject_dirs, case_entries)
            return committed
        manifest = brats_patient_split(case_entries, self.subject_dirs, seed=_SPLIT_SEED)
        save_committed_split(
            "brats_wt", manifest,
            provenance=("BraTS-GLI patient_v2: all sessions of each five-digit patient "
                        "stay together; stdlib random.Random(42), sorted patients, "
                        "rounded 20% test. Per-session top-10 seg>0 axial slices "
                        "with at least 64 foreground pixels; val empty."),
        )
        return manifest

    def __len__(self) -> int:
        return len(self.case_ids)

    # Keep at most two decoded subjects in memory per process.
    @lru_cache(maxsize=2)
    def _load_subject_slices(
        self, sub_id: str
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Decode the pinned raw volumes."""
        import nibabel as nib  # local: heavy optional dep

        sub_dir = self.subject_dirs[sub_id]
        flair_vol = nib.load(str(sub_dir / f"{sub_id}-t2f.nii.gz")).get_fdata()
        seg_vol = nib.load(str(sub_dir / f"{sub_id}-seg.nii.gz")).get_fdata()
        kept = self._pick_slices(seg_vol)
        K = len(kept)
        H, W = int(seg_vol.shape[0]), int(seg_vol.shape[1])
        img_u8 = np.zeros((K, H, W), dtype=np.uint8)
        seg_bool = np.zeros((K, H, W), dtype=bool)
        for k, axi in enumerate(kept):
            img_slice = np.asarray(flair_vol[:, :, axi], dtype=np.float32)
            lo, hi = float(img_slice.min()), float(img_slice.max())
            if hi > lo:
                img_u8[k] = (
                    (img_slice - lo) / (hi - lo) * 255.0
                ).astype(np.uint8)
            else:
                img_u8[k] = 0
            seg_bool[k] = np.asarray(seg_vol[:, :, axi]) > 0

        return img_u8, seg_bool, kept

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        case_id = self.case_ids[idx]
        sub_id, axi = self._slice_index[case_id]

        img_u8_stack, seg_bool_stack, kept = self._load_subject_slices(sub_id)
        try:
            k = kept.index(int(axi))
        except ValueError as exc:
            raise RuntimeError(
                f"[brats] axi={axi} not in cached slices for {sub_id}: {kept}"
            ) from exc
        img_u8 = img_u8_stack[k]
        mask_bool = seg_bool_stack[k]

        img_t = torch.from_numpy(np.ascontiguousarray(img_u8)).unsqueeze(0).repeat(3, 1, 1)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_bool)).unsqueeze(0)

        img_t = _resize_uint8(img_t, _TARGET_HW)
        mask_t = _resize_bool(mask_t, _TARGET_HW)

        if self.transform is not None:
            img_t, mask_t = self.transform(img_t, mask_t)
        return img_t, mask_t, case_id


def _resize_uint8(img: torch.Tensor, size: int) -> torch.Tensor:
    import torch.nn.functional as F

    x = img.to(torch.float32).unsqueeze(0)
    x = F.interpolate(
        x, size=(size, size), mode="bilinear", align_corners=False
    )
    return x.clamp(0, 255).squeeze(0).to(torch.uint8)


def _resize_bool(mask: torch.Tensor, size: int) -> torch.Tensor:
    x = mask.to(torch.float32).unsqueeze(0)
    x = resize_mask(x, (size, size))  # nearest unless FMPOOL_MASK_RESIZE=nearest-exact
    return x.squeeze(0).to(torch.bool)
