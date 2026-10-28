#!/usr/bin/env python3
"""Train UNet-skip decoder over 4 multi-scale FM features (one seed).

Relation to the 1x1 read-out
----------------------------
The primary read-out is a 1x1-conv decoder over the FINAL feature map. The UNet-skip read-out deliberately
swaps that for a UNet decoder with skip connections to multi-scale FM
features (see ``src/fmpool/multiscale.py`` and
``src/fmpool/heads_unet.py``). This script is the trainer for that probe.

Pipeline
--------
1. Build the frozen multi-scale encoder (4 levels of features per image).
2. Build the :class:`UNetSkipHead` decoder over the encoder's level dims.
3. Train online (without a feature cache): each batch runs the frozen encoder under
   ``torch.no_grad()`` in FP32 (TF32 off; bf16/AMP requests are rejected), then the decoder forward + BCE loss.
   Online is preferred over a multi-scale cache because for ResNet50 and
   ConvNeXt the largest level is 56x56 -- caching 4 levels at fp16 would
   eat several GB per (task, fm) combination.
4. Evaluate per-case Dice on the test split using the same pipeline.
5. Atomically write checkpoint + per-case JSON.

Output path:
    {out_root}/{task}/{fm}/seed_{seed}.json

JSON schema (unet_skip_v1):
    task / fm / seed / head_design = "unet_skip"
    n_trainable_params, n_test, test_ids, per_case_dice, mean_dice
    head_checkpoint_sha256, training_elapsed_s, use_amp_bf16
    decoder_channels, level_dims (list[int] of len 4), level_hw
    schema_version = "unet_skip_v1"

Exit codes
----------
0 success, 2 FM unloadable, 3 dataset unavailable.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fmpool import datasets as fmpool_datasets
from fmpool import provenance
from fmpool import determinism, multiscale
from fmpool.heads_unet import build_unet_skip_head, count_trainable_params

logger = logging.getLogger("fmpool.train_head_unet_skip")

INPUT_HW: tuple[int, int] = (224, 224)
LR: float = 1e-3
EPOCHS_DEFAULT: int = 30
BATCH_SIZE_DEFAULT: int = 16
DECODER_CHANNELS_DEFAULT: int = 128
SCHEMA_VERSION: str = "unet_skip_v1"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class _RejectLowPrecision(argparse.Action):
    """``--amp``/``--bf16`` fail loudly; they add nothing to the namespace (run identity unchanged)."""

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(f"{option_string}: this trainer runs FP32 only (TF32 off); "
                     "bf16/AMP is not supported")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UNet-skip head trainer")
    p.add_argument("--task", required=True)
    p.add_argument("--fm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output root (e.g. results/per_case_dice_unet); subdir tree appended.",
    )
    p.add_argument("--checkpoint-dir", type=Path, required=True,
                   help="directory for the seed_<S>.pt decoder checkpoint")
    p.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument(
        "--decoder-channels", type=int, default=DECODER_CHANNELS_DEFAULT
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", "--bf16", nargs=0, action=_RejectLowPrecision, default=argparse.SUPPRESS,
                   help="Rejected with an error: bf16/AMP is not supported (FP32 only).")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    provenance.add_recipe_args(p, primary=False)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Atomic IO
# ---------------------------------------------------------------------------


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    try:
        json.dumps(json.loads(data), allow_nan=False)
    except ValueError as exc:
        raise determinism.NumericalHealthError("nonfinite or invalid score JSON") from exc
    if path.exists():
        raise FileExistsError(f"preserving existing artifact: {path}")
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
    determinism.require_finite(obj, "saved checkpoint")
    if path.exists():
        raise FileExistsError(f"preserving existing artifact: {path}")
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


# ---------------------------------------------------------------------------
# Data plumbing (mirrors train_adapted.py)
# ---------------------------------------------------------------------------


def _make_normaliser(
    mean: tuple[float, float, float], std: tuple[float, float, float]
) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def _fn(image: torch.Tensor, mask: torch.Tensor):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"image expected [3,H,W]; got {tuple(image.shape)}")
        img = image.float()
        if image.dtype == torch.uint8 or img.max() > 1.5:
            img = img / 255.0
        img = F.interpolate(
            img.unsqueeze(0), size=INPUT_HW, mode="bilinear", align_corners=False
        ).squeeze(0)
        img = (img - mean_t) / std_t
        m = mask
        if m.ndim == 2:
            m = m.unsqueeze(0)
        if m.shape[-2:] != INPUT_HW:
            m = F.interpolate(
                m.unsqueeze(0).float(), size=INPUT_HW, mode="nearest"
            ).squeeze(0)
        return img, m.to(torch.bool if mask.dtype == torch.bool else mask.dtype)

    return _fn


def _resize_mask_to_input(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    resized = F.interpolate(
        mask.float().unsqueeze(0), size=INPUT_HW, mode="nearest"
    ).squeeze(0)
    return (resized > 0.5).float()


def _collate(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack(
        [_resize_mask_to_input(item[1]) for item in batch], dim=0
    )
    ids = [item[2] for item in batch]
    return imgs, masks, ids


def _seed_worker(worker_id: int) -> None:
    base = torch.initial_seed() % (2**32)
    np.random.seed(base + worker_id)
    import random as _r

    _r.seed(base + worker_id)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def _train(
    encoder: nn.Module,
    fm_id: str,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> None:
    determinism.require_finite(encoder.state_dict(), "encoder parameters")
    determinism.require_finite(head.state_dict(), "head parameters")
    head.train()
    encoder.eval()
    opt = torch.optim.Adam(
        [p for p in head.parameters() if p.requires_grad], lr=LR
    )
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        running = 0.0
        n_batches = 0
        for imgs, masks, _ids in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                feats = multiscale.extract_multiscale_features(
                    encoder, fm_id, imgs
                )
                feats = [f.detach().float() for f in feats]
            determinism.require_finite(feats, "features")
            logits = head(feats)
            determinism.require_finite(logits, "logits")
            loss = bce(logits.float(), masks.float())
            determinism.checked_step(loss, opt)
            running += float(loss.item())
            n_batches += 1
        avg = running / max(n_batches, 1)
        if (epoch + 1) % max(epochs // 5, 1) == 0 or epoch == 0:
            logger.info("epoch %d/%d loss=%.4f", epoch + 1, epochs, avg)


@torch.no_grad()
def _evaluate(
    encoder: nn.Module,
    fm_id: str,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    determinism.require_finite(encoder.state_dict(), "encoder parameters")
    determinism.require_finite(head.state_dict(), "head parameters")
    encoder.eval()
    head.eval()
    from fmpool.estimators import dice_per_case

    all_ids: list[str] = []
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for imgs, masks, ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks_np = (masks.numpy() > 0.5).astype(bool)
        feats = multiscale.extract_multiscale_features(encoder, fm_id, imgs)
        determinism.require_finite(feats, "features")
        logits = head([f.float() for f in feats])
        determinism.require_finite(logits, "logits")
        preds = (torch.sigmoid(logits.float()).cpu().numpy() > 0.5).astype(bool)
        all_ids.extend(ids)
        all_preds.append(preds.reshape(preds.shape[0], -1))
        all_targets.append(masks_np.reshape(masks_np.shape[0], -1))
    if not all_preds:
        return [], np.zeros((0,), dtype=np.float64)
    pred_all = np.concatenate(all_preds, axis=0)
    tgt_all = np.concatenate(all_targets, axis=0)
    per_case = dice_per_case(pred_all, tgt_all)
    return all_ids, per_case


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    with determinism.numerical_failure_record(
        Path(args.out) / args.task / args.fm / f"seed_{args.seed}",
        task=args.task, fm=args.fm, stage="train_head_unet_skip", seed=args.seed,
    ):
        return _run(args)


def _run(args: argparse.Namespace) -> int:
    inputs = provenance.read_input_manifest(args.input_manifest, args.task)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    determinism.set_seed(args.seed)

    if args.data_root:
        os.environ["FMPOOL_DATA_ROOT"] = str(args.data_root)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    try:
        encoder, ms_spec = multiscale.build_multiscale_encoder(args.fm)
    except determinism.NumericalHealthError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("FM %s unloadable: %s", args.fm, exc)
        print(
            f"[train_head_unet_skip] ERROR: cannot load FM {args.fm!r}: {exc}",
            file=sys.stderr,
        )
        return 2
    encoder = encoder.float().to(device)
    provenance.verify_encoder_pin(encoder, inputs, required=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    normalise_fn = _make_normaliser(ms_spec.norm_mean, ms_spec.norm_std)
    try:
        train_ds = fmpool_datasets.build_dataset(
            args.task, split="train", transform=normalise_fn
        )
        test_ds = fmpool_datasets.build_dataset(
            args.task, split="test", transform=normalise_fn
        )
    except (FileNotFoundError, NotImplementedError) as exc:
        logger.error("dataset unavailable: %s", exc)
        return 3

    provenance.verify_dataset_inputs(inputs, train_ds, test_ds)
    determinism.require_finite(encoder.state_dict(), "initial checkpoint")
    initial_state_sha = provenance.preserve_initial_state(
        encoder, Path(args.out), inputs)
    determinism.set_seed(args.seed)

    loader_kwargs: dict = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    if args.num_workers > 0:
        loader_kwargs["worker_init_fn"] = _seed_worker
    train_loader = DataLoader(
        train_ds, sampler=determinism.SeededShuffle(train_ds, args.seed), drop_last=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, shuffle=False, drop_last=False, **loader_kwargs
    )

    head = build_unet_skip_head(
        level_dims=ms_spec.level_dims,
        num_classes=1,
        out_size=INPUT_HW,
        decoder_channels=args.decoder_channels,
    ).to(device)
    n_params = count_trainable_params(head)
    logger.info(
        "UNet-skip head: fm=%s level_dims=%s decoder_ch=%d n_trainable=%d",
        args.fm,
        ms_spec.level_dims,
        args.decoder_channels,
        n_params,
    )

    t0 = time.time()
    _train(
        encoder=encoder,
        fm_id=args.fm,
        head=head,
        loader=train_loader,
        device=device,
        epochs=args.epochs,
    )
    train_elapsed = time.time() - t0

    test_ids, per_case = _evaluate(
        encoder=encoder,
        fm_id=args.fm,
        head=head,
        loader=test_loader,
        device=device,
    )

    out_root = Path(args.out)
    out_dir = out_root / args.task / args.fm
    out_path = out_dir / f"seed_{args.seed}.json"

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_path = ckpt_dir / f"seed_{args.seed}.pt"
    _atomic_torch_save(head.state_dict(), ckpt_path)
    ckpt_sha = determinism.sha256_file(ckpt_path)

    payload = {
        "run_identity_v2": provenance.run_identity(args, __file__, inputs=inputs,
            initial_state_sha256=initial_state_sha,
            recipe={"version": "secondary_fp32_v2", "optimizer": "Adam", "lr": LR}),
        "task": args.task,
        "fm": args.fm,
        "seed": int(args.seed),
        "head_design": "unet_skip",
        "decoder_channels": int(args.decoder_channels),
        "level_dims": list(ms_spec.level_dims),
        "level_hw": [list(hw) for hw in ms_spec.level_hw],
        "n_trainable_params": int(n_params),
        "n_test": int(per_case.size),
        "test_ids": list(test_ids),
        "per_case_dice": [float(x) for x in per_case.tolist()],
        "mean_dice": float(np.mean(per_case)) if per_case.size else float("nan"),
        "head_checkpoint_sha256": ckpt_sha,
        "training_elapsed_s": float(train_elapsed),
        "use_amp_bf16": False,  # FP32 only; key kept for the result schema
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "schema_version": SCHEMA_VERSION,
    }
    body = json.dumps(payload, indent=2).encode("utf-8")
    _atomic_write_bytes(out_path, body)

    logger.info(
        "wrote %s n_test=%d mean_dice=%.4f train_s=%.1f",
        out_path,
        payload["n_test"],
        payload["mean_dice"],
        train_elapsed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
