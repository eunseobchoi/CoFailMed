#!/usr/bin/env python3
"""Train a 1x1-conv decoder on cached features (one seed per call).

Reads ``<cache-root>/{task}/{fm}/{train,test}.pt`` produced by
``extract_features.py`` and trains a fresh ``LinearSegHead`` for the given
seed. Per-case Dice on the test split is written to a JSON file with the same
schema as ``train_frozen_fm.py``.

Optimizations vs. ``train_frozen_fm.py``:
  - encoder forward eliminated (cached features amortise across seeds + epochs)
  - FP32 head + BCE (AMP off)
  - on-device per-case Dice (no host transfer of mask tensors)
  - on-device loss accumulation (one ``.item()`` per epoch, not per step)

Determinism: fmpool.determinism.set_seed before head init, deterministic
shuffle via per-seed generator. Determinism checks still apply.

Exit codes
----------
0 success, 3 cache missing/invalid, 4 manifest mismatch.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

from fmpool import determinism, provenance
from fmpool.decoder import LinearSegHead

logger = logging.getLogger("fmpool.train_head_cached")

INPUT_HW: tuple[int, int] = (224, 224)
LR: float = 1e-3
EPOCHS_DEFAULT: int = 30
BATCH_SIZE_DEFAULT: int = 16
SCHEMA_VERSION: str = provenance.CACHE_SCHEMA


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FP32 head trainer (cached features)")
    p.add_argument("--task", required=True)
    p.add_argument("--fm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--cache-root", type=Path, required=True,
                   help="feature cache root written by extract_features.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True,
                   help="directory for the seed_<S>.pt head checkpoint")
    p.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    provenance.add_recipe_args(p, cached=True, primary=True)
    return p.parse_args(argv)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
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
    determinism.require_finite(obj, "saved tensors")
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


def _load_cache(cache_dir: Path, *, allow_legacy=False, expected_sha256=None):
    return provenance.load_feature_cache(cache_dir, allow_legacy=allow_legacy,
                                         expected_sha256=expected_sha256)


def _per_case_dice_on_device(
    preds: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    """Compute per-case Dice on-device.

    preds, masks: bool [B, H, W]. Returns float32 [B] tensor.
    Conventions match fmpool.estimators.dice_per_case (both empty -> 1.0).
    """
    p = preds.flatten(1).to(torch.float32)
    m = masks.flatten(1).to(torch.float32)
    inter = (p * m).sum(dim=1)
    denom = p.sum(dim=1) + m.sum(dim=1)
    dice = torch.where(
        denom > 0, 2.0 * inter / denom, torch.ones_like(inter)
    )
    return dice


def _train_head(
    head: LinearSegHead,
    feats: torch.Tensor,
    masks: torch.Tensor,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> None:
    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()
    n = feats.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    for epoch in range(epochs):
        idx = torch.randperm(n, generator=gen)
        running = torch.zeros((), device=device, dtype=torch.float32)
        n_batches = 0
        for start in range(0, n, batch_size):
            end = start + batch_size
            chunk = idx[start:end]
            f_b = feats[chunk].to(device, non_blocking=True)
            m_b = masks[chunk].to(device, non_blocking=True).to(torch.float32).unsqueeze(1)
            determinism.require_finite(f_b, "training features")
            opt.zero_grad(set_to_none=True)
            logits = head(f_b.to(torch.float32))
            loss = bce(logits, m_b)
            determinism.require_finite(logits, "training logits")
            determinism.checked_step(loss, opt)
            running += loss.detach().to(torch.float32)
            n_batches += 1
        avg = float(running.item()) / max(n_batches, 1)
        if (epoch + 1) % max(epochs // 5, 1) == 0 or epoch == 0:
            logger.info("epoch %d/%d loss=%.4f", epoch + 1, epochs, avg)


@torch.no_grad()
def _evaluate_head(
    head: LinearSegHead,
    feats: torch.Tensor,
    masks: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    head.eval()
    n = feats.shape[0]
    case_dices: list[torch.Tensor] = []
    for start in range(0, n, batch_size):
        end = start + batch_size
        f_b = feats[start:end].to(device, non_blocking=True)
        m_b = masks[start:end].to(device, non_blocking=True).to(torch.bool)
        determinism.require_finite(f_b, "evaluation features")
        logits = head(f_b.to(torch.float32))
        determinism.require_finite(logits, "evaluation logits")
        preds = (torch.sigmoid(logits.float()) > 0.5).squeeze(1)
        case_dices.append(_per_case_dice_on_device(preds, m_b).cpu())
    if not case_dices:
        return torch.zeros((0,), dtype=torch.float32)
    return torch.cat(case_dices, dim=0)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    with determinism.numerical_failure_record(
        Path(args.out) / f"seed_{args.seed}", task=args.task, fm=args.fm,
        stage="train_head_cached", seed=getattr(args, "seed", None),
    ):
        return _run(args)


def _run(args: argparse.Namespace) -> int:
    provenance.validate_primary_recipe(args)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    if not args.allow_legacy_cache and not args.cache_manifest_sha256:
        raise ValueError("--cache-manifest-sha256 is required for v2 training")
    determinism.set_seed(args.seed)

    cache_root = Path(args.cache_root)
    cache_dir = cache_root / args.task / args.fm
    try:
        manifest, train_blob, test_blob = _load_cache(cache_dir, allow_legacy=args.allow_legacy_cache,
                                                      expected_sha256=args.cache_manifest_sha256)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("cache invalid: %s", exc)
        return 3

    if manifest.get("task") != args.task or manifest.get("fm") != args.fm:
        logger.error(
            "manifest mismatch: file says (%s,%s) but called with (%s,%s)",
            manifest.get("task"), manifest.get("fm"), args.task, args.fm,
        )
        return 4

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    feature_dim = int(manifest["feature_dim"])
    feat_h, feat_w = manifest["feature_hw"]

    # Keep cached tensors on CPU (pinned for fast async H2D); move only the
    # current minibatch to GPU. Avoids OOM on BraTS-scale caches (~5 GB).
    pin = (device.type == "cuda")
    train_feats = train_blob["feats"].pin_memory() if pin else train_blob["feats"]
    train_masks = train_blob["masks"].pin_memory() if pin else train_blob["masks"]
    test_feats = test_blob["feats"].pin_memory() if pin else test_blob["feats"]
    test_masks = test_blob["masks"].pin_memory() if pin else test_blob["masks"]
    test_ids = list(test_blob["case_ids"])

    if train_feats.shape[1] != feature_dim or train_feats.shape[2] != feat_h:
        logger.error(
            "feature shape mismatch with manifest: tensor %s vs manifest D=%d hw=%dx%d",
            tuple(train_feats.shape), feature_dim, feat_h, feat_w,
        )
        return 4

    head = LinearSegHead(
        in_dim=feature_dim, num_classes=1, out_size=INPUT_HW
    ).to(device)

    t0 = time.time()
    _train_head(
        head=head,
        feats=train_feats,
        masks=train_masks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    train_elapsed = time.time() - t0

    per_case = _evaluate_head(
        head=head,
        feats=test_feats,
        masks=test_masks,
        batch_size=args.batch_size,
        device=device,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_path = ckpt_dir / f"seed_{args.seed}.pt"
    _atomic_torch_save(head.state_dict(), ckpt_path)
    ckpt_sha = determinism.sha256_file(ckpt_path)

    manifest_path = cache_dir / "manifest.json"
    cache_manifest_sha = determinism.sha256_file(manifest_path)

    out_dir = Path(args.out)
    out_path = out_dir / f"seed_{args.seed}.json"
    payload = {
        "run_identity_v2": provenance.run_identity(args, __file__, cache=manifest,
            recipe=provenance.PRIMARY_RECIPE),
        "task": args.task,
        "fm": args.fm,
        "seed": int(args.seed),
        "n_test": int(per_case.numel()),
        "test_ids": list(test_ids),
        "per_case_dice": [float(x) for x in per_case.tolist()],
        "mean_dice": float(per_case.mean().item()) if per_case.numel() else float("nan"),
        "cache_manifest_sha256": cache_manifest_sha,
        "head_checkpoint_sha256": ckpt_sha,
        "training_elapsed_s": float(train_elapsed),
        "use_amp_bf16": False,  # FP32 only; key kept for the result schema
        "schema_version": SCHEMA_VERSION,
    }
    body = json.dumps(payload, indent=2).encode("utf-8")
    _atomic_write_bytes(out_path, body)

    logger.info(
        "wrote %s n_test=%d mean_dice=%.4f train_s=%.1f",
        out_path, payload["n_test"], payload["mean_dice"], train_elapsed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
