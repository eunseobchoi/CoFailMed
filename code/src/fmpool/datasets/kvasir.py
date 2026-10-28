"""Kvasir-SEG polyp segmentation loader (task ``kvasir``).

Binary mask via ``mask > 127`` (JPEG lossy). The primary official 880/120
split is the default (``FMPOOL_KVASIR_SPLIT=official``); ``FMPOOL_KVASIR_SPLIT=grouped_v2``
selects the content-group-disjoint split used for the appendix sensitivity.
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
    kvasir_split_name,
    load_committed_split,
    require_dir,
    resolve_task_root,
    save_committed_split,
)

_EXPECTED_TOTAL = 1000
_TRAIN_COUNT = 880
_TEST_COUNT = 120
_MASK_BINARISE_THRESHOLD = 127

# Sorted, newline-terminated IDs from Kvasir-SEG@6f375457/Data-split/{train,val}.txt.
_OFFICIAL_MEMBERSHIP_SHA256 = {
    "train": "268f069ffdbbcb3aa5a6ac11827793600fb7bef4202a72a14d404927940fdab8",
    "test": "bbd395e7980afbd801269628f79fdffcb47401575b30e84e1668d89e157c998a",
}


# Declared near-duplicate pair (64x64 grayscale image correlation r = 0.998461). The content rule
# below also detects it, so the generated content-group manifest lists this pair twice.
_KNOWN_NEAR_DUPLICATE_GROUPS = (
    ("cju886ryxnsl50801r93jai7q", "cju8amfdtqi4x09871tygrgqe"),
)


def _grouped_split(stems, groups) -> dict:
    stems = sorted(stems)
    if len(stems) != _EXPECTED_TOTAL or len(set(stems)) != len(stems):
        raise ValueError("Kvasir requires 1000 unique case IDs")
    units = {cid: {cid} for cid in stems}
    for group in groups:
        present = set(group) & set(stems)
        merged = set().union(*(units[cid] for cid in present))
        for cid in merged:
            units[cid] = merged
    unique = {tuple(sorted(unit)) for unit in units.values()}
    ordered = sorted(unique, key=lambda unit: hashlib.sha256(
        ("kvasir_grouped_v2:" + ",".join(unit)).encode()).hexdigest())
    def take(count, remaining):
        reachable = {0: []}
        for unit in remaining:
            for n, selected in list(reachable.items()):
                total = n + len(unit)
                if total <= count and total not in reachable:
                    reachable[total] = selected + [unit]
        if count not in reachable:
            raise ValueError("declared groups cannot form requested split sizes")
        selected = reachable[count]
        return {c for unit in selected for c in unit}, [u for u in remaining if u not in selected]

    test, remaining = take(_TEST_COUNT, ordered)
    return {"train": sorted(c for unit in remaining for c in unit),
            "val": [], "test": sorted(test),
            "split_schema_version": "kvasir_grouped_v2",
            "split_scheme": "primary_locked_880_120_no_selection",
            "test_role": "retrospective_evaluation",
            "near_duplicate_groups": sorted([sorted(g) for g in groups])}


def _content_groups(images_dir: Path, masks_dir: Path, stems) -> tuple[list, dict]:
    """Fixed content rule, independent of model outcomes; transitive groups."""
    from fmpool.determinism import sha256_file

    images, masks, hashes = [], [], {}
    for cid in sorted(stems):
        image_path, mask_path = images_dir / f"{cid}.jpg", masks_dir / f"{cid}.jpg"
        with Image.open(image_path) as image:
            images.append(np.asarray(image.convert("L").resize((64, 64)), dtype=np.float64).ravel())
        with Image.open(mask_path) as mask:
            masks.append(np.asarray(mask.convert("L").resize((64, 64), Image.Resampling.NEAREST)).ravel() > 127)
        hashes[cid] = [sha256_file(image_path), sha256_file(mask_path)]
    x = np.stack(images)
    x -= x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x /= np.maximum(norm, 1e-12)
    correlations = x @ x.T
    groups = list(_KNOWN_NEAR_DUPLICATE_GROUPS)
    exact = {}
    for cid, (image_hash, _) in hashes.items():
        exact.setdefault(image_hash, []).append(cid)
    groups.extend(tuple(ids) for ids in exact.values() if len(ids) > 1)
    stems = sorted(stems)
    for i, j in zip(*np.where(np.triu(correlations >= 0.995, k=1))):
        denominator = masks[i].sum() + masks[j].sum()
        dice = 2 * (masks[i] & masks[j]).sum() / max(denominator, 1)
        if dice >= 0.95:
            groups.append((stems[i], stems[j]))
    return groups, hashes


def _check_official_split(split: dict, stems, groups, hashes, counts=(_TRAIN_COUNT, _TEST_COUNT)) -> None:
    """Official 880/120 lists: pinned raw content, exact partition, detected groups wholly in train."""
    train, test = set(split["train"]), set(split["test"])
    if (len(split["train"]), len(split["test"])) != counts or split["val"]:
        raise RuntimeError("Kvasir official split must be 880/120 with an empty val list")
    if train & test or train | test != set(stems):
        raise RuntimeError("Kvasir official split does not partition the case IDs")
    for role, expected in _OFFICIAL_MEMBERSHIP_SHA256.items():
        actual = hashlib.sha256(("\n".join(sorted(split[role])) + "\n").encode()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Kvasir official {role} membership differs from the pinned source lists")
    if split["content_sha256"] != hashes:
        raise RuntimeError("Kvasir raw content differs from the official split manifest")
    detected = sorted({tuple(sorted(g)) for g in groups})
    if (not all(set(g) <= train for g in detected) or split["near_duplicate_groups_all_in_train"] is not True
            or [list(g) for g in detected] != split["near_duplicate_groups"]):
        raise RuntimeError("Kvasir near-duplicate group not wholly in official train (or not the declared groups)")


class KvasirSEG(Dataset):
    """Kvasir-SEG polyp segmentation.

    Returns ``(image_uint8 [3,H,W], mask_bool [1,H,W], case_id)``.
    No augmentation or normalisation here — the training script
    applies the FM-specific normalisation.
    """

    def __init__(
        self,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Kvasir split must be train/val/test, got {split!r}"
            )
        self.split = split
        self.transform = transform
        self.root = resolve_task_root("kvasir")
        require_dir(self.root, dataset="kvasir")

        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        require_dir(self.images_dir, dataset="kvasir")
        require_dir(self.masks_dir, dataset="kvasir")

        image_files = sorted(
            p.name for p in self.images_dir.iterdir()
            if p.suffix.lower() == ".jpg"
        )
        if len(image_files) != _EXPECTED_TOTAL:
            raise RuntimeError(
                f"[kvasir] expected {_EXPECTED_TOTAL} images in "
                f"{self.images_dir}, found {len(image_files)}"
            )

        self.case_ids: list[str] = self._load_or_generate_split(image_files)[
            split
        ]

    def _load_or_generate_split(
        self, image_files: list[str]
    ) -> dict[str, list[str]]:
        stems = [Path(f).stem for f in image_files]
        groups, hashes = _content_groups(self.images_dir, self.masks_dir, stems)
        if kvasir_split_name() == "kvasir_official":  # never generated; the committed file is the source
            official = load_committed_split("kvasir")
            if official is None:
                raise FileNotFoundError("kvasir_official_split.json is missing")
            _check_official_split(official, stems, groups, hashes)
            return official
        expected = _grouped_split(stems, groups)
        expected["content_sha256"] = hashes
        expected["content_scan"] = {"images": len(stems), "pairs": len(stems) * (len(stems) - 1) // 2}
        expected["group_rule"] = "gray64_corr>=0.995_and_mask64_dice>=0.95_plus_audit_pair"
        committed = load_committed_split("kvasir_grouped_v2")
        if committed is not None:
            if any(committed.get(k) != v for k, v in expected.items()):
                raise RuntimeError("Kvasir grouped v2 manifest mismatch; preserve and inspect it")
            return committed
        save_committed_split("kvasir_grouped_v2", expected,
                             provenance="Content-group split: groups ordered by SHA-256; the declared near-duplicate pair is kept together")
        return expected

    def __len__(self) -> int:
        return len(self.case_ids)

    def _resolve(self, case_id: str, where: Path) -> Path:
        candidate = where / case_id
        if candidate.is_file():
            return candidate
        candidate_jpg = where / f"{case_id}.jpg"
        if candidate_jpg.is_file():
            return candidate_jpg
        raise FileNotFoundError(
            f"[kvasir] file not found for case_id={case_id!r} in {where}"
        )

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        case_id = self.case_ids[idx]
        img_path = self._resolve(case_id, self.images_dir)
        mask_path = self._resolve(case_id, self.masks_dir)

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

        return image_t, mask_t, Path(case_id).stem
