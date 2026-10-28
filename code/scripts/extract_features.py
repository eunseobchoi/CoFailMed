#!/usr/bin/env python3
"""Extract and cache encoder features once per (task, fm).

Reads the committed train/test split for ``task``, runs the frozen FM
encoder in FP32 once over both splits, and writes
``[N, D, h, w]`` FP32 feature tensors plus 224x224 uint8 mask tensors and
the case_ids into ``<cache-root>/{task}/{fm}/{train,test}.pt``.

Subsequent cached-head trainers only need ``torch.load`` of these tensors
and a 1x1 conv decoder, eliminating per-seed/per-epoch encoder forwards.

Cache invariants are recorded in ``manifest.json`` so cached-head trainers can validate
provenance. Features are deterministic given the same (encoder weights,
data, normalisation).

Exit codes
----------
0 success, 2 FM unloadable, 3 dataset unavailable, 4 cache validation fail.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fmpool import datasets as fmpool_datasets
from fmpool import determinism, encoders, provenance
from fmpool.provenance import cache_identity_matches as _cache_identity_matches

logger = logging.getLogger("fmpool.extract_features")

INPUT_HW: tuple[int, int] = (224, 224)
SCHEMA_VERSION: str = provenance.CACHE_SCHEMA


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FP32 feature extractor")
    p.add_argument("--task", required=True)
    p.add_argument("--fm", required=True)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--cache-root", type=Path, required=True,
                   help="feature cache root; the cache is written to <root>/<task>/<fm>")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    provenance.add_recipe_args(p)
    return p.parse_args(argv)


def _make_normaliser(spec: encoders.EncoderSpec) -> Callable:
    mean = torch.tensor(spec.norm_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(spec.norm_std, dtype=torch.float32).view(3, 1, 1)

    def _fn(image: torch.Tensor, mask: torch.Tensor):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"image expected [3,H,W], got {tuple(image.shape)}")
        img = image.float()
        if image.dtype == torch.uint8 or img.max() > 1.5:
            img = img / 255.0
        img = F.interpolate(
            img.unsqueeze(0), size=INPUT_HW, mode="bilinear", align_corners=False
        ).squeeze(0)
        img = (img - mean) / std
        m = mask
        if m.ndim == 2:
            m = m.unsqueeze(0)
        if m.shape[-2:] != INPUT_HW:
            m = fmpool_datasets.resize_mask(m.unsqueeze(0).float(), INPUT_HW).squeeze(0)
        return img, m.to(torch.bool if mask.dtype == torch.bool else mask.dtype)

    return _fn


def _collate(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack(
        [item[1].squeeze(0) if item[1].ndim == 3 else item[1] for item in batch],
        dim=0,
    )  # [B, 224, 224]
    ids = [item[2] for item in batch]
    return imgs, masks, ids


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch_save(obj, path: Path) -> None:
    determinism.require_finite(obj, "saved tensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


@torch.no_grad()
def _extract_split(
    encoder, fm_id: str, loader: DataLoader, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str], int, int]:
    feats_chunks: list[torch.Tensor] = []
    masks_chunks: list[torch.Tensor] = []
    ids_all: list[str] = []
    feat_d = -1
    feat_h = -1
    for imgs, masks, ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        feats = encoders.extract_features(encoder, fm_id, imgs)
        feats_fp32 = feats.to(torch.float32).contiguous().cpu()
        determinism.require_finite(feats_fp32, "extracted FP32 features")
        feats_chunks.append(feats_fp32)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        masks_u8 = masks.to(torch.bool).to(torch.uint8).cpu()
        masks_chunks.append(masks_u8)
        ids_all.extend(ids)
        if feat_d < 0:
            feat_d = feats_fp32.shape[1]
            feat_h = feats_fp32.shape[2]
    feats_all = torch.cat(feats_chunks, dim=0)
    masks_all = torch.cat(masks_chunks, dim=0)
    if feats_all.shape[0] != len(ids_all) or masks_all.shape[0] != len(ids_all):
        raise RuntimeError(
            f"extract: shape mismatch feats {feats_all.shape} masks "
            f"{masks_all.shape} ids {len(ids_all)}"
        )
    return feats_all, masks_all, ids_all, feat_d, feat_h


def _cache_dir(cache_root: Path, task: str, fm: str) -> Path:
    return Path(cache_root) / task / fm


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    with determinism.numerical_failure_record(
        Path(args.cache_root) / args.task / args.fm, task=args.task, fm=args.fm,
        stage="extract_features", seed=getattr(args, "seed", None),
    ):
        return _run(args)


def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    determinism.set_seed(0)

    if args.data_root:
        os.environ["FMPOOL_DATA_ROOT"] = str(args.data_root)

    cache_root = Path(args.cache_root)
    out_dir = _cache_dir(cache_root, args.task, args.fm)
    manifest_path = out_dir / "manifest.json"
    train_path = out_dir / "train.pt"
    test_path = out_dir / "test.pt"

    inputs = provenance.read_input_manifest(args.input_manifest, args.task)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    try:
        encoder, spec = encoders.build_encoder(args.fm)
    except Exception as exc:
        logger.error("FM %s unloadable: %s", args.fm, exc)
        return 2
    encoder = encoder.float().to(device)
    determinism.require_finite(encoder.state_dict(), "encoder checkpoint")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    normalise_fn = _make_normaliser(spec)

    try:
        train_ds = fmpool_datasets.build_dataset(args.task, "train", transform=normalise_fn)
        test_ds = fmpool_datasets.build_dataset(args.task, "test", transform=normalise_fn)
    except (FileNotFoundError, NotImplementedError) as exc:
        logger.error("dataset unavailable: %s", exc)
        return 3

    provenance.verify_dataset_inputs(inputs, train_ds, test_ds)
    source = provenance.source_snapshot()
    runtime = provenance.runtime_identity()
    declared_source = inputs["manifest"].get("encoder_source")
    if not declared_source:
        raise ValueError("input manifest requires encoder_source (repository/revision/checkpoint)")
    if any(not declared_source.get(key) for key in ("repository", "revision", "state_sha256")):
        raise ValueError("encoder_source requires repository, revision, state_sha256")
    if provenance.state_sha256(encoder) != declared_source["state_sha256"]:
        raise ValueError("loaded encoder state differs from pinned input manifest")
    encoder_source = provenance.encoder_source(encoder, declared_source)
    expected = {
        "schema_version": SCHEMA_VERSION, "task": args.task, "fm": args.fm,
        "precision": args.precision,
        "split_manifest_sha256": inputs["manifest"]["split_manifest_sha256"],
        "ground_truth_identity": provenance.ground_truth_identity(args.task),
        "preprocess_identity": provenance.preprocess_identity(INPUT_HW, spec.norm_mean, spec.norm_std),
        "code_sha256": provenance.digest(source),
        "encoder_checkpoint_sha256": provenance.state_sha256(encoder),
        "input_manifest_sha256": inputs["sha256"],
        "runtime_sha256": provenance.digest(runtime),
        "encoder_source_sha256": provenance.digest(encoder_source),
    }
    if out_dir.exists():
        try:
            old, _, _ = provenance.load_feature_cache(out_dir)
            if not _cache_identity_matches(old, expected):
                raise ValueError("cache identity mismatch")
        except (OSError, ValueError, KeyError) as exc:
            logger.error("preserving existing cache; choose a fresh --cache-root: %s", exc)
            return 4
        logger.info("verified identical v2 cache: %s", out_dir)
        return 0

    def _make_loader(ds):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
            collate_fn=_collate,
            drop_last=False,
        )

    train_loader = _make_loader(train_ds)
    test_loader = _make_loader(test_ds)

    t0 = time.time()
    train_feats, train_masks, train_ids, fd, fh = _extract_split(
        encoder, args.fm, train_loader, device
    )
    train_elapsed = time.time() - t0
    logger.info(
        "train: N=%d D=%d HxW=%dx%d %.1fs",
        train_feats.shape[0], fd, fh, train_feats.shape[3], train_elapsed,
    )

    t1 = time.time()
    test_feats, test_masks, test_ids, _, _ = _extract_split(
        encoder, args.fm, test_loader, device
    )
    test_elapsed = time.time() - t1
    logger.info("test: N=%d %.1fs", test_feats.shape[0], test_elapsed)

    out_dir.mkdir(parents=True, exist_ok=False)
    _atomic_torch_save(
        {"feats": train_feats, "masks": train_masks, "case_ids": train_ids},
        train_path,
    )
    _atomic_torch_save(
        {"feats": test_feats, "masks": test_masks, "case_ids": test_ids},
        test_path,
    )

    _atomic_torch_save(encoder.state_dict(), out_dir / "encoder_state.pt")
    _atomic_write_bytes(out_dir / "provenance.json", json.dumps({
        "source_snapshot": source, "runtime": runtime, "inputs": inputs,
        "encoder_source": encoder_source,
        "riga_training_palette": getattr(train_ds, "palette_audit", None),
        "riga_test_palette": getattr(test_ds, "palette_audit", None),
    }, sort_keys=True).encode())

    manifest = {
        **expected,
        "identity_sha256": provenance.digest(expected),
        "artifact_sha256": {name: determinism.sha256_file(out_dir / name) for name in
            ("train.pt", "test.pt", "encoder_state.pt", "provenance.json")},
        "task": args.task,
        "fm": args.fm,
        "feature_dim": int(spec.feature_dim),
        "feature_hw": [int(train_feats.shape[2]), int(train_feats.shape[3])],
        "input_hw": list(INPUT_HW),
        "use_amp_bf16": False,  # FP32 only; key read by load_feature_cache
        "n_train": int(train_feats.shape[0]),
        "n_test": int(test_feats.shape[0]),
        "train_foreground_pixels": train_masks.flatten(1).sum(1).tolist(),
        "test_foreground_pixels": test_masks.flatten(1).sum(1).tolist(),
        "split_scheme": inputs["split"].get("split_scheme", "committed_task_split"),
        "riga_training_palette_verified": bool(getattr(train_ds, "palette_audit", None)),
        "train_case_ids_sha256": provenance.digest(train_ids),
        "test_case_ids_sha256": provenance.digest(test_ids),
        "norm_mean": list(spec.norm_mean),
        "norm_std": list(spec.norm_std),
        "extract_seconds": float(train_elapsed + test_elapsed),
    }
    _atomic_write_bytes(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=False).encode("utf-8"),
    )
    logger.info("wrote cache to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
