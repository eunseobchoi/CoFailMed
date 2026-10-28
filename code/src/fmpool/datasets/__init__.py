"""Medical segmentation dataset loaders.

Public API:
    build_dataset(task, split, transform=None) -> torch.utils.data.Dataset
    list_splits(task) -> dict[str, list[str]]

All loaders return ``(image, mask, case_id)`` where ``image`` is a
``uint8`` tensor in ``[0, 255]`` of shape ``[3, H, W]`` (normalisation
is the training script's job) and ``mask`` is a ``bool``
tensor of shape ``[1, H, W]``.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import torch
import yaml
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_CODE_ROOT = Path(__file__).resolve().parents[3]
_PATHS_YAML = _CODE_ROOT / "configs" / "paths.yaml"
_SPLITS_DIR = _CODE_ROOT / "data" / "splits"

_DEFAULT_DATA_ROOT = str(_CODE_ROOT / "data" / "raw")

_VALID_TASKS: tuple[str, ...] = (
    "kvasir",
    "acdc_lv",
    "brats_wt",
    "riga_cup",
    "riga_disc",
    "msd_hippocampus",
    "isic2018",
    "msd_heart",
    "msd_spleen",
)


Transform = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _load_paths_config() -> dict:
    """Load configs/paths.yaml if present; return {} otherwise."""
    if not _PATHS_YAML.is_file():
        return {}
    with _PATHS_YAML.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    return loaded or {}


def resolve_data_root() -> Path:
    """Resolve the raw-data root.

    Priority: ``FMPOOL_DATA_ROOT`` env var > ``configs/paths.yaml``
    > repository-local default path. Does NOT require the directory to
    exist — callers are responsible for emitting a descriptive
    ``FileNotFoundError`` via :func:`require_dir`.
    """
    env = os.environ.get("FMPOOL_DATA_ROOT", "").strip()
    if env:
        return Path(env)
    cfg = _load_paths_config()
    root = Path(cfg.get("data_root", _DEFAULT_DATA_ROOT))
    if not root.is_absolute():
        root = _CODE_ROOT / root
    return root


def resolve_task_root(task: str) -> Path:
    """Return the dataset directory for ``task`` under the data root."""
    cfg = _load_paths_config()
    subs = (cfg.get("datasets", {}) if cfg else {}) or {}
    default_sub = {
        "kvasir": "Kvasir-SEG",
        "acdc": "ACDC",
        "brats": "brats2024_gli",
        "riga": "RIGA/RIGAPlus",
        "msd_hippocampus": "Task04_Hippocampus",
        "msd_heart": "Task02_Heart",
        "msd_spleen": "Task09_Spleen",
        "isic2018": "ISIC2018_official",
    }[task]
    if task == "isic2018":
        isic_split_name()  # Validate the requested split before resolving its raw root.
    return resolve_data_root() / subs.get(task, default_sub)


def require_dir(path: Path, *, dataset: str) -> None:
    """Raise a descriptive ``FileNotFoundError`` if ``path`` is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"[{dataset}] expected data directory not found: {path}\n"
            f"  Set FMPOOL_DATA_ROOT to a directory containing the dataset, "
            f"or edit {_PATHS_YAML} (default root: {_DEFAULT_DATA_ROOT})."
        )


# ---------------------------------------------------------------------------
# Split manifest persistence
# ---------------------------------------------------------------------------


# The paper's primary splits are the defaults; the other entries reproduce sensitivity or superseded runs.
_KVASIR_SPLITS = {"": "kvasir_official", "official": "kvasir_official", "grouped_v2": "kvasir_grouped_v2"}
_ACDC_SPLITS = {"": "acdc_official", "official": "acdc_official", "cv5": "acdc_lv"}
_ISIC_SPLITS = {"": "isic2018_official", "official": "isic2018_official"}


def _env_choice(var: str, table: dict) -> str:
    choice = os.environ.get(var, "").strip()
    if choice not in table:
        raise ValueError(f"{var} must be one of {sorted(table)}, got {choice!r}")
    return table[choice]


def acdc_split_name() -> str:
    """``acdc_official`` (default or ``FMPOOL_ACDC_SPLIT=official``: train = the 100 official training patients,
    test = patients 101-150) or ``acdc_lv`` (``FMPOOL_ACDC_SPLIT=cv5``: fold 0 of a superseded seed-42 5-fold CV)."""
    return _env_choice("FMPOOL_ACDC_SPLIT", _ACDC_SPLITS)


def isic_split_name() -> str:
    """Official Task 1 training/test split (default or ``FMPOOL_ISIC_SPLIT=official``)."""
    return _env_choice("FMPOOL_ISIC_SPLIT", _ISIC_SPLITS)


def kvasir_split_name() -> str:
    """Committed Kvasir split: the official Jha et al. 2021 880/120 lists (default or ``FMPOOL_KVASIR_SPLIT=official``;
    the study's primary split) or the content-group-disjoint sensitivity split (``FMPOOL_KVASIR_SPLIT=grouped_v2``)."""
    choice = os.environ.get("FMPOOL_KVASIR_SPLIT", "").strip()
    if choice not in _KVASIR_SPLITS:
        raise ValueError(f"FMPOOL_KVASIR_SPLIT must be one of {sorted(_KVASIR_SPLITS)}, got {choice!r}")
    return _KVASIR_SPLITS[choice]


_MASK_RESIZE_MODES = {"": "nearest", "nearest": "nearest", "nearest-exact": "nearest-exact"}


def mask_resize_mode() -> str:
    """Mask resize mode: ``nearest`` (default, floor(i*s) sampling, ~0.5 output px off the bilinear
    image grid) or ``nearest-exact`` (pixel-centre aligned) via ``FMPOOL_MASK_RESIZE`` (sensitivity only)."""
    choice = os.environ.get("FMPOOL_MASK_RESIZE", "").strip()
    if choice not in _MASK_RESIZE_MODES:
        raise ValueError(f"FMPOOL_MASK_RESIZE must be one of {sorted(_MASK_RESIZE_MODES)}, got {choice!r}")
    return _MASK_RESIZE_MODES[choice]


def resize_mask(x: torch.Tensor, size) -> torch.Tensor:
    """Resize a float ``[N, C, H, W]`` mask with :func:`mask_resize_mode` (shared by the 2D loaders and normalisers)."""
    import torch.nn.functional as F

    return F.interpolate(x, size=size, mode=mask_resize_mode())


def splits_path(task: str) -> Path:
    """Path to the committed split JSON for ``task``."""
    if task == "kvasir":
        task = kvasir_split_name()
    if task == "acdc_lv":
        task = acdc_split_name()
    if task == "isic2018":
        task = isic_split_name()
    if task == "brats_wt":
        task = "brats_wt_patient_v2"
    filename_task = {"kvasir_grouped_v2": "kvasir_grouped", "brats_wt_patient_v2": "brats_patient"}.get(task, task)
    return _SPLITS_DIR / f"{filename_task}_split.json"


def load_committed_split(task: str) -> Optional[dict]:
    """Return the committed split JSON dict, or ``None`` if absent."""
    p = splits_path(task)
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_committed_split(task: str, split: dict, *, provenance: str) -> None:
    """Persist the split JSON with a provenance note."""
    _SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"task": task, "provenance": provenance, **split}
    with splits_path(task).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)


# ---------------------------------------------------------------------------
# Public dataset builders
# ---------------------------------------------------------------------------


def build_dataset(
    task: str,
    split: str,
    transform: Optional[Transform] = None,
) -> Dataset:
    """Instantiate a dataset for ``(task, split)``.

    Supported tasks: ``kvasir``, ``acdc_lv``, ``brats_wt``,
    ``riga_cup``, ``riga_disc``, ``msd_hippocampus``, ``isic2018``,
    ``msd_heart``, ``msd_spleen``. Supported splits: ``train`` / ``val``
    / ``test`` (``val`` is empty in the committed manifests); ACDC additionally accepts
    ``fold_{k}_train`` / ``fold_{k}_val`` for 5-fold CV.
    """
    if task not in _VALID_TASKS:
        raise KeyError(
            f"Unknown task {task!r}; expected one of {sorted(_VALID_TASKS)}"
        )
    # Import only the module actually needed — acdc/brats pull in
    # nibabel, which is not installed on dev machines without the
    # medical-imaging extras.
    if task == "kvasir":
        from fmpool.datasets.kvasir import KvasirSEG

        return KvasirSEG(split=split, transform=transform)
    if task == "acdc_lv":
        from fmpool.datasets.acdc import ACDCLVDataset

        return ACDCLVDataset(split=split, transform=transform)
    if task == "brats_wt":
        from fmpool.datasets.brats import BraTSWTDataset

        return BraTSWTDataset(split=split, transform=transform)
    if task in {"riga_cup", "riga_disc"}:
        from fmpool.datasets.riga import RIGADataset

        target = "cup" if task == "riga_cup" else "disc"
        return RIGADataset(target=target, split=split, transform=transform)
    if task == "msd_hippocampus":
        from fmpool.datasets.msd_hippocampus import MSDHippocampusDataset

        return MSDHippocampusDataset(split=split, transform=transform)
    if task == "msd_heart":
        from fmpool.datasets.msd_heart import MSDHeartDataset

        return MSDHeartDataset(split=split, transform=transform)
    if task == "msd_spleen":
        from fmpool.datasets.msd_spleen import MSDSpleenDataset

        return MSDSpleenDataset(split=split, transform=transform)
    if task == "isic2018":
        from fmpool.datasets.isic2018 import ISIC2018Dataset

        return ISIC2018Dataset(split=split, transform=transform)
    raise ValueError(f"unknown task: {task!r}")


def list_splits(task: str) -> dict[str, list[str]]:
    """Return the committed split dict for ``task``.

    Does not trigger split generation. The official Kvasir, ACDC and ISIC-2018
    splits are read-only; the other loaders write their split file on the first
    successful ``build_dataset`` call when none is committed.
    """
    committed = load_committed_split(task)
    if committed is None:
        raise FileNotFoundError(
            f"split JSON for task {task!r} has not been committed yet; "
            f"instantiate the dataset once to generate {splits_path(task)}"
        )
    return {k: committed[k] for k in ("train", "val", "test") if k in committed}


__all__ = [
    "Transform",
    "build_dataset",
    "list_splits",
    "acdc_split_name",
    "isic_split_name",
    "mask_resize_mode",
    "resize_mask",
    "load_committed_split",
    "require_dir",
    "resolve_data_root",
    "resolve_task_root",
    "save_committed_split",
    "splits_path",
]
