"""ISIC 2018 Task 1 binary skin-lesion segmentation loader.

The default task root is ``ISIC2018_official``. Training uses the 2,594 images
in ``ISIC2018_Task1-2_Training_Input`` with ``ISIC2018_Task1_Training_GroundTruth``;
testing uses the 1,000 images in ``ISIC2018_Task1-2_Test_Input`` with
``ISIC2018_Task1_Test_GroundTruth``. Masks are binarised via ``uint8 > 127``.

Only the committed ``isic2018_official_split.json`` is used; on-disk paired
cases must equal its lists. Splits are never generated or rewritten here.
``FMPOOL_ISIC_SPLIT`` may be unset or ``official``; other values are rejected.
Patient identifiers are partially available from the ISIC Archive, but this
loader does not use them to group the split. The official partitions share
known patient and lesion IDs; this image-level benchmark does not establish
patient-disjoint generalisation. Missing patient identifiers prevent a
complete overlap assessment.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from fmpool.datasets import (
    load_committed_split,
    require_dir,
    resolve_task_root,
)

_MASK_BINARISE_THRESHOLD = 127

_INPUT_SUBDIR = "ISIC2018_Task1-2_Training_Input"
_GT_SUBDIR = "ISIC2018_Task1_Training_GroundTruth"
OFFICIAL_DIRS = {"train": (_INPUT_SUBDIR, _GT_SUBDIR),
                 "test": ("ISIC2018_Task1-2_Test_Input", "ISIC2018_Task1_Test_GroundTruth")}
OFFICIAL_SCHEME = "official_isic2018_task1_train_test"
_OFFICIAL_COUNTS = {"train": 2594, "test": 1000}
# Sorted, newline-terminated IDs from code/data/splits/isic2018_official_split.json.
_OFFICIAL_MEMBERSHIP_SHA256 = {
    "train": "b6a11fa50b1cf0363160258b5cac96ccae382ea524e34434c3212d1fbb1e4775",
    "test": "6fb4df9f361085d70966ce9098133a6491b756869717af4eaef4ca687686e4f7",
}


def validate_isic_official_split(manifest: dict) -> None:
    """Require the complete, disjoint Task 1 cohort pinned by the committed official listing."""
    if manifest.get("split_scheme") != OFFICIAL_SCHEME or manifest.get("val") != []:
        raise ValueError("[isic2018] official split requires the official scheme and an empty val")
    for role, count in _OFFICIAL_COUNTS.items():
        ids = manifest[role]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError(f"[isic2018] official {role} requires {count} unique cases, found {len(ids)}")
    if set(manifest["train"]) & set(manifest["test"]):
        raise ValueError("[isic2018] official train/test lists must be disjoint")
    for role, expected in _OFFICIAL_MEMBERSHIP_SHA256.items():
        actual = hashlib.sha256(("\n".join(sorted(manifest[role])) + "\n").encode()).hexdigest()
        if actual != expected:
            raise ValueError(f"[isic2018] official {role} membership differs from the pinned committed listing")


class ISIC2018Dataset(Dataset):
    """ISIC 2018 Task 1 skin-lesion binary segmentation.

    Returns ``(image_uint8 [3,H,W], mask_bool [1,H,W], case_id)`` where
    ``case_id`` is the ISIC stem (e.g. ``ISIC_0000000``). Native image
    resolution is preserved (downstream resize is the training script's
    job, matching kvasir.py behaviour).
    """

    def __init__(
        self,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"isic2018 split must be train/val/test, got {split!r}"
            )
        self.split = split
        self.transform = transform
        self.root = resolve_task_root("isic2018")
        require_dir(self.root, dataset="isic2018")

        role = split if split == "test" else "train"
        images_sub, masks_sub = OFFICIAL_DIRS[role]
        self.images_dir = self.root / images_sub
        self.masks_dir = self.root / masks_sub
        require_dir(self.images_dir, dataset="isic2018")
        require_dir(self.masks_dir, dataset="isic2018")

        manifest = self._load_official_split()
        self.case_ids: list[str] = list(manifest[split])

    @staticmethod
    def _list_jpgs(directory: Path) -> list[str]:
        out: list[str] = []
        for p in sorted(directory.iterdir()):
            if not p.is_file():
                continue
            name = p.name
            if name.startswith("._"):
                continue
            if p.suffix.lower() != ".jpg":
                continue
            out.append(name)
        return out

    def _discover_cases(self) -> list[str]:
        image_files = self._list_jpgs(self.images_dir)
        cases: list[str] = []
        for fname in image_files:
            stem = Path(fname).stem  # ISIC_0000000
            mask_path = self.masks_dir / f"{stem}_segmentation.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"[isic2018] official mask missing for {fname}: {mask_path}")
            cases.append(stem)
        mask_ids = {p.name.removesuffix("_segmentation.png") for p in self.masks_dir.glob("*_segmentation.png")
                    if p.is_file() and not p.name.startswith("._")}
        if mask_ids != set(cases):
            raise ValueError("[isic2018] official masks without matching images")
        return cases

    def _load_official_split(self) -> dict:
        """Committed official lists only; the paired cases on disk for this split's folders must equal them."""
        committed = load_committed_split("isic2018")  # -> isic2018_official_split.json
        if committed is None:
            raise FileNotFoundError("isic2018_official_split.json is missing (FMPOOL_ISIC_SPLIT=official)")
        if committed.get("split_scheme") != OFFICIAL_SCHEME or committed.get("val") != []:
            raise RuntimeError("[isic2018] committed official split has the wrong scheme or a non-empty val")
        validate_isic_official_split(committed)
        role = "test" if self.split == "test" else "train"
        on_disk = self._discover_cases()
        if on_disk != list(committed[role]):
            raise RuntimeError(f"[isic2018] paired cases on disk for {role} differ from the committed official list "
                               f"({len(on_disk)} vs {len(committed[role])})")
        return {"train": list(committed["train"]), "val": [], "test": list(committed["test"])}

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        case_id = self.case_ids[idx]
        img_path = self.images_dir / f"{case_id}.jpg"
        mask_path = self.masks_dir / f"{case_id}_segmentation.png"

        image = np.array(
            Image.open(img_path).convert("RGB"), dtype=np.uint8, copy=True
        )
        image_t = torch.from_numpy(image).permute(2, 0, 1).contiguous()

        mask_raw = np.asarray(
            Image.open(mask_path).convert("L"), dtype=np.uint8
        )
        mask_bool = mask_raw > _MASK_BINARISE_THRESHOLD
        mask_t = torch.from_numpy(mask_bool).unsqueeze(0)

        if self.transform is not None:
            image_t, mask_t = self.transform(image_t, mask_t)
        return image_t, mask_t, case_id
