"""RIGA+ fundus disc/cup loader (tasks ``riga_cup`` and ``riga_disc``).

Expected on-disk layout:
  <task_root>/BinRushed_{train,test}.csv
  <task_root>/Magrabia_{train,test}.csv
  <task_root>/MESSIDOR_Base{1,2,3}_{train,test}.csv
  <task_root>/RIGA/<source>/<subdir>/imageN.tif
  <task_root>/RIGA-mask/<source>/<subdir>/imageN-{1..6}.tif

Each rater TIFF is single-channel uint8 with values {0, 128, 255}:
0 = background, 128 = filled cup, 255 = disc rim. Disc
binary = rater >= 128. Majority vote across 6 raters
(``sum(rater_i) >= 4``).

Train = BinRushed + MESSIDOR, test = Magrabia. We
pool the *train* and *test* CSVs within each source so the full
BinRushed / MESSIDOR / Magrabia complements are used (the per-file
sub-splits inside each source are unused here).
"""
from __future__ import annotations

import csv
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from fmpool.datasets import (
    load_committed_split,
    resize_mask,
    require_dir,
    resolve_task_root,
    save_committed_split,
)

_NUM_RATERS = 6
_MAJORITY_THRESHOLD = 4
_TARGET_HW = 224
_CUP_PIXEL = 128
GROUND_TRUTH_IDENTITY = "riga_cup128_disc_union_vote4of6_v2"
_DISC_MIN_PIXEL = 128

_TRAIN_SOURCES = (
    "BinRushed",
    "MESSIDOR_Base1",
    "MESSIDOR_Base2",
    "MESSIDOR_Base3",
)
_TEST_SOURCES = ("Magrabia",)


def _target_pixels(raw: np.ndarray, target: str) -> np.ndarray:
    if raw.ndim != 2 or not np.isin(raw, (0, 128, 255)).all():
        raise ValueError("unsupported RIGA raw mask palette")
    if target == "cup":
        return raw == _CUP_PIXEL
    if target == "disc":
        return raw >= _DISC_MIN_PIXEL
    raise ValueError(f"unknown RIGA target: {target}")


def require_training_palette_verification(split: str, audit: dict | None) -> None:
    if split != "train":
        return
    if not audit or any(audit.get(k) != v for k, v in {
        "schema_version": "riga_palette_audit_v1", "cup_pixel": 128,
        "disc_min_pixel": 128, "all_training_masks_checked": True,
    }.items()):
        raise RuntimeError("RIGA training palette must be verified from raw masks")
    cases, hashes = audit.get("training_case_ids", []), audit.get("mask_sha256", {})
    if not cases or len(hashes) != 6 * len(cases):
        raise RuntimeError("RIGA training palette audit must cover all six raw raters per case")


class RIGADataset(Dataset):
    """RIGA+ fundus cup/disc (tasks ``riga_cup`` / ``riga_disc``).

    Returns ``(image_uint8 [3,224,224], mask_bool [1,224,224], case_id)``
    with ``case_id`` equal to the CSV image path minus its leading
    ``RIGA/`` prefix (unique across the train+test pool).
    """

    def __init__(
        self,
        target: str,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        if target not in {"cup", "disc"}:
            raise ValueError(
                f"RIGA target must be 'cup' or 'disc', got {target!r}"
            )
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"RIGA split must be train/val/test, got {split!r}"
            )
        self.target = target
        self.split = split
        self.transform = transform
        self.root = resolve_task_root("riga")
        require_dir(self.root, dataset="riga")

        task_name = f"riga_{target}"
        manifest = self._load_or_generate_split(task_name)
        from fmpool.provenance import validate_source_split
        validate_source_split(task_name, manifest)
        self._image_paths: dict[str, str] = manifest["_image_paths"]
        self.case_ids: list[str] = list(manifest[split])
        self.palette_audit = self._verify_training_palette() if split in {"train", "test"} else None
        require_training_palette_verification(split, self.palette_audit)

    def _read_csv_rows(
        self, sources: Iterable[str]
    ) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for src in sources:
            for sub in ("train", "test"):
                csv_path = self.root / f"{src}_{sub}.csv"
                if not csv_path.is_file():
                    continue
                with csv_path.open("r", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        img_rel = row["image"].strip()
                        mask_rel = row["mask"].strip()
                        if img_rel and mask_rel:
                            rows.append((img_rel, mask_rel))
        return rows

    def _load_or_generate_split(self, task_name: str) -> dict:
        committed = load_committed_split(task_name)
        if committed is not None:
            return {
                "train": list(committed.get("train", [])),
                "val": list(committed.get("val", [])),
                "test": list(committed.get("test", [])),
                "_image_paths": dict(committed.get("_image_paths", {})),
            }

        train_rows = self._read_csv_rows(_TRAIN_SOURCES)
        test_rows = self._read_csv_rows(_TEST_SOURCES)
        if not train_rows or not test_rows:
            raise RuntimeError(
                f"[riga] empty CSV pool: train={len(train_rows)}, "
                f"test={len(test_rows)}. Expected BinRushed + MESSIDOR "
                f"and Magrabia CSVs under {self.root}"
            )

        image_paths: dict[str, str] = {}
        train_ids: list[str] = []
        test_ids: list[str] = []
        for img_rel, _mask_rel in train_rows:
            cid = self._case_id_for(img_rel)
            image_paths[cid] = img_rel
            train_ids.append(cid)
        for img_rel, _mask_rel in test_rows:
            cid = self._case_id_for(img_rel)
            image_paths[cid] = img_rel
            test_ids.append(cid)

        provenance = (
            "RIGA+ official CSVs: train = BinRushed + MESSIDOR_Base1-3 "
            "(pooled both _train and _test CSVs per source); test = "
            "Magrabia (pooled). Majority vote across 6 rater TIFFs: "
            "cup = sum(rater==128) ≥ 4, disc = sum(rater>=128) ≥ 4."
        )
        persisted = {
            "train": train_ids,
            "val": [],
            "test": test_ids,
            "_image_paths": image_paths,
        }
        save_committed_split(task_name, persisted, provenance=provenance)
        return {
            "train": train_ids,
            "val": [],
            "test": test_ids,
            "_image_paths": image_paths,
        }

    @staticmethod
    def _case_id_for(image_rel: str) -> str:
        if image_rel.startswith("RIGA/"):
            return image_rel[len("RIGA/"):]
        return image_rel

    def _rater_mask_paths(self, image_rel: str) -> list[Path]:
        img_path = Path(image_rel)
        stem = img_path.stem  # "image1"
        rel_parent = img_path.parent
        # Strip leading "RIGA" component so mask dir mirrors under RIGA-mask/
        parts = rel_parent.parts
        if parts and parts[0] == "RIGA":
            parts = parts[1:]
        mask_dir = self.root / "RIGA-mask" / Path(*parts)
        return [
            mask_dir / f"{stem}-{i}.tif"
            for i in range(1, _NUM_RATERS + 1)
        ]

    def _verify_training_palette(self) -> dict:
        """Reuse only complete audits bound to freshly hashed raw inputs."""
        cache_root = os.environ.get("FMPOOL_RIGA_AUDIT_CACHE")
        if not cache_root:
            return self._audit_training_palette()
        from fmpool.determinism import sha256_file
        from fmpool.provenance import digest
        import PIL
        import scipy

        root, cache = self.root.resolve(), Path(cache_root).resolve()
        if cache == root or root in cache.parents:
            raise ValueError("RIGA audit cache must be outside the raw root")
        if not self.case_ids or len(set(self.case_ids)) != len(self.case_ids):
            raise RuntimeError("RIGA training palette: empty or duplicate training cases")
        masks = []
        for cid in self.case_ids:
            paths = self._rater_mask_paths(self._image_paths[cid])
            if len(paths) != _NUM_RATERS:
                raise ValueError("RIGA audit requires six raters per case")
            for path in paths:
                masks.append([str(path.relative_to(self.root)),
                              str(path.resolve(strict=True).relative_to(root)), sha256_file(path)])
        hashes = {rel: sha for rel, resolved, sha in masks}
        if len(hashes) != _NUM_RATERS * len(self.case_ids):
            raise ValueError("RIGA audit has duplicate mask paths")
        identity = {
            "schema_version": "riga_audit_cache_v1",
            "ground_truth_identity": GROUND_TRUTH_IDENTITY,
            "algorithm_sha256": digest(inspect.getsource(type(self)._audit_training_palette)),
            "libraries": {"numpy": np.__version__, "scipy": scipy.__version__,
                          "pillow": PIL.__version__},
            "case_ids": list(self.case_ids), "split": self.split, "masks": masks,
        }
        path = cache / (digest(identity) + ".json")

        def validate(record):
            try:
                result = record["result"]
                expected = {
                    "schema_version": "riga_palette_audit_v1", "cup_pixel": 128,
                    "disc_min_pixel": 128, "all_training_masks_checked": self.split == "train",
                    "training_case_ids": list(self.case_ids), "mask_sha256": hashes,
                    "split": self.split, "topology": result["topology"],
                }
                if (set(record) != {"identity", "result", "sha256"}
                        or record["identity"] != identity or digest(result) != digest(expected)
                        or record["sha256"] != digest({"identity": identity, "result": result})
                        or set(result["topology"]) != set(hashes)):
                    raise ValueError("content/identity mismatch")
                for stats in result["topology"].values():
                    if (set(stats) != {"cup_pixels", "disc_pixels", "cup_hole_pixels",
                                      "disc_hole_pixels", "cup_enclosed_by_rim"}
                            or any(type(stats[k]) is not int for k in
                                   ("cup_pixels", "disc_pixels", "cup_hole_pixels", "disc_hole_pixels"))
                            or not 0 < stats["cup_pixels"] < stats["disc_pixels"]
                            or stats["cup_hole_pixels"] != 0 or stats["disc_hole_pixels"] != 0
                            or type(stats["cup_enclosed_by_rim"]) is not bool):
                        raise ValueError("invalid topology")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid RIGA audit cache: {path}") from exc
            return result

        if path.exists():
            return validate(json.loads(path.read_text()))
        result = self._audit_training_palette()
        # Bind decoding to the same bytes hashed before and after the cold audit.
        if result["mask_sha256"] != hashes or any(
                sha256_file(self.root / rel) != sha for rel, resolved, sha in masks):
            raise ValueError("RIGA raw masks changed during audit")
        record = {"identity": identity, "result": result}
        record["sha256"] = digest(record)
        validate(record)
        cache.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".riga-", dir=cache)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(record, stream, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(name, path)  # Atomic publication; never replace another writer's record.
            except FileExistsError:
                validate(json.loads(path.read_text()))
        finally:
            Path(name).unlink()
        return result

    def _audit_training_palette(self) -> dict:
        """Read every raw training rater; never trust a stale audit flag."""
        from fmpool.determinism import sha256_file
        from scipy.ndimage import binary_fill_holes

        checked = {}
        topology = {}
        if not self.case_ids or len(set(self.case_ids)) != len(self.case_ids):
            raise RuntimeError("RIGA training palette: empty or duplicate training cases")
        for cid in self.case_ids:
            for path in self._rater_mask_paths(self._image_paths[cid]):
                with Image.open(path) as image:
                    raw = np.asarray(image)
                if raw.ndim != 2 or set(np.unique(raw)) != {0, 128, 255}:
                    raise RuntimeError(f"RIGA training palette unsupported: {path}")
                cup, disc = raw == 128, raw >= 128
                cup_holes = int((binary_fill_holes(cup) & ~cup).sum())
                disc_holes = int((binary_fill_holes(disc) & ~disc).sum())
                if cup_holes or disc_holes:
                    raise RuntimeError(f"RIGA training palette topology requires review: {path}")
                rel = str(path.relative_to(self.root))
                checked[rel] = sha256_file(path)
                topology[rel] = {"cup_pixels": int(cup.sum()), "disc_pixels": int(disc.sum()),
                    "cup_hole_pixels": cup_holes, "disc_hole_pixels": disc_holes,
                    "cup_enclosed_by_rim": bool(np.all(binary_fill_holes(raw == 255)[cup]))}
        return {
            "schema_version": "riga_palette_audit_v1",
            "cup_pixel": 128, "disc_min_pixel": 128,
            "all_training_masks_checked": self.split == "train",
            # historical key name: the case IDs of this split (train or test)
            "training_case_ids": list(self.case_ids), "mask_sha256": checked,
            "split": self.split, "topology": topology,
        }

    def _load_majority_mask(self, image_rel: str) -> np.ndarray:
        rater_paths = self._rater_mask_paths(image_rel)
        votes: Optional[np.ndarray] = None
        n_found = 0
        for p in rater_paths:
            if not p.is_file():
                continue
            rater = np.asarray(
                Image.open(p).convert("L"), dtype=np.uint8
            )
            hit = _target_pixels(rater, self.target)
            votes = (
                hit.astype(np.uint8)
                if votes is None
                else votes + hit.astype(np.uint8)
            )
            n_found += 1
        if votes is None or n_found == 0:
            raise FileNotFoundError(
                f"[riga] no rater masks found for {image_rel} under "
                f"{self.root / 'RIGA-mask'}"
            )
        if n_found != _NUM_RATERS:
            # A vote over fewer than six raters would change the 4-of-6 rule.
            raise FileNotFoundError(
                f"[riga] expected {_NUM_RATERS} rater masks for {image_rel}, "
                f"found {n_found}. Cannot compute the 4-of-6 majority vote."
            )
        return votes >= _MAJORITY_THRESHOLD

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        case_id = self.case_ids[idx]
        image_rel = self._image_paths[case_id]
        img_path = self.root / image_rel
        image = np.array(
            Image.open(img_path).convert("RGB"), dtype=np.uint8, copy=True
        )
        image_t = torch.from_numpy(image).permute(2, 0, 1).contiguous()

        mask_bool = self._load_majority_mask(image_rel)
        mask_t = torch.from_numpy(mask_bool).unsqueeze(0)

        image_t = _resize_uint8(image_t, _TARGET_HW)
        mask_t = _resize_bool(mask_t, _TARGET_HW)

        if self.transform is not None:
            image_t, mask_t = self.transform(image_t, mask_t)
        return image_t, mask_t, case_id


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
