#!/usr/bin/env python3
"""Train the fine-tuning and from-scratch encoder controls with a 1x1 head.

``full_ft`` unfreezes the pretrained encoder; ``rand_init`` trains the encoder
and head from random initialisation. Both use FP32 Adam and BCE.
Traces are written to ``{out}/{regime}/{config}/{task}/{fm}/seed_{seed}.json``,
where config is ``ft_e{epochs}`` or ``rand_init``. Fine-tuning learning-rate arms
use separate output roots; the head learning rate remains 1e-3.
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
from fmpool import determinism, encoders
from fmpool.decoder import LinearSegHead

logger = logging.getLogger("fmpool.train_adapted")

INPUT_HW: tuple[int, int] = (224, 224)
LR: float = 1e-3
# ---------------------------------------------------------------------------
# Random-init encoders (regime ``rand_init``)
# ---------------------------------------------------------------------------


def build_random_encoder(fm: str) -> tuple[nn.Module, encoders.EncoderSpec]:
    """Rebuild the FM architecture WITHOUT pretrained weights."""
    spec = encoders.FM_SPECS[fm]
    if fm.startswith("dinov2_"):
        wrapper: nn.Module = encoders._load_dinov2(fm, pretrained=False)
    elif fm == "biomedclip":
        import open_clip

        model = open_clip.create_model(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            pretrained=None,
            load_weights=False,
            pretrained_text=False,
        )
        visual = model.visual
        trunk = visual.trunk if hasattr(visual, "trunk") else visual
        wrapper = encoders._BiomedCLIPWrapper(trunk, spec.feature_dim, spec.feature_hw)
    elif fm == "clip_vitl14":
        import open_clip

        model = open_clip.create_model("ViT-L-14-quickgelu", pretrained=None)
        for m in model.visual.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        wrapper = encoders._CLIPViTLWrapper(model.visual, spec.feature_dim, spec.feature_hw)
    elif fm in ("convnext_tiny", "efficientnet_b0", "resnet50", "resnet18"):
        import torchvision.models as tvm

        ctor = {
            "convnext_tiny": tvm.convnext_tiny,
            "efficientnet_b0": tvm.efficientnet_b0,
            "resnet50": tvm.resnet50,
            "resnet18": tvm.resnet18,
        }[fm]
        model = ctor(weights=None)
        trunk = nn.Sequential(*list(model.children())[:-2])
        wrapper = encoders._CNNWrapper(trunk, spec.feature_dim, spec.feature_hw)
    else:
        raise KeyError(f"Unknown fm {fm!r}")
    return wrapper, spec


# ---------------------------------------------------------------------------
# Training helpers matching the frozen-encoder input and scoring conventions
# ---------------------------------------------------------------------------


def make_normaliser(
    spec: encoders.EncoderSpec,
) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    mean = torch.tensor(spec.norm_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(spec.norm_std, dtype=torch.float32).view(3, 1, 1)

    def _fn(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"normalise_fn expects [3,H,W], got {tuple(image.shape)}")
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
            m = F.interpolate(
                m.unsqueeze(0).float(), size=INPUT_HW, mode="nearest"
            ).squeeze(0)
        return img, m.to(torch.bool if mask.dtype == torch.bool else mask.dtype)

    return _fn


def _resize_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    resized = F.interpolate(
        mask.float().unsqueeze(0), size=INPUT_HW, mode="nearest"
    ).squeeze(0)
    return (resized > 0.5).float()


def _collate(batch: list[tuple[torch.Tensor, torch.Tensor, str]]):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([_resize_mask(item[1]) for item in batch], dim=0)
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


def _make_optimizer(
    encoder: nn.Module, decoder: nn.Module, encoder_lr: float | None = None
) -> torch.optim.Adam:
    """Adam over trainable encoder + decoder params.

    ``encoder_lr=None``: one group at ``LR`` (original recipe). Otherwise the
    encoder params form their own group at ``encoder_lr``; the head keeps ``LR``.
    """
    enc = [p for p in encoder.parameters() if p.requires_grad]
    dec = [p for p in decoder.parameters() if p.requires_grad]
    if not enc + dec:
        raise RuntimeError("no trainable parameters in encoder + decoder")
    if encoder_lr is None:
        return torch.optim.Adam(enc + dec, lr=LR)
    return torch.optim.Adam([{"params": enc, "lr": encoder_lr}, {"params": dec, "lr": LR}], lr=LR)


def _train(
    encoder: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    encoder_train_mode: bool,
    encoder_lr: float | None = None,
) -> None:
    determinism.require_finite(encoder.state_dict(), "encoder parameters")
    determinism.require_finite(decoder.state_dict(), "decoder parameters")
    opt = _make_optimizer(encoder, decoder, encoder_lr)
    loss_fn = nn.BCEWithLogitsLoss()

    encoder.train() if encoder_train_mode else encoder.eval()
    decoder.train()

    for epoch in range(epochs):
        n_batches = 0
        running = 0.0
        for imgs, masks, _ids in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            feats = encoder(imgs)
            determinism.require_finite(feats, "features")
            logits = decoder(feats)
            determinism.require_finite(logits, "logits")
            loss = loss_fn(logits.float(), masks.float())
            opt.zero_grad(set_to_none=True)
            determinism.checked_step(loss, opt)
            running += float(loss.item())
            n_batches += 1
        logger.info(
            "epoch %d/%d loss=%.4f", epoch + 1, epochs, running / max(n_batches, 1)
        )


@torch.no_grad()
def _evaluate(
    encoder: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    determinism.require_finite(encoder.state_dict(), "encoder parameters")
    determinism.require_finite(decoder.state_dict(), "decoder parameters")
    encoder.eval()
    decoder.eval()
    from fmpool.estimators import dice_per_case

    all_ids: list[str] = []
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for imgs, masks, ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks_np = (masks.numpy() > 0.5).astype(bool)
        feats = encoder(imgs)
        determinism.require_finite(feats, "features")
        logits = decoder(feats)
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


def _atomic_save(state: dict[str, torch.Tensor], path: Path) -> None:
    determinism.require_finite(state, "saved checkpoint")
    if path.exists():
        raise FileExistsError(f"preserving existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tuning and from-scratch control trainer.")
    p.add_argument("--task", required=True)
    p.add_argument("--fm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--regime", choices=("full_ft", "rand_init"), required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Per-case-Dice ROOT (e.g. results/per_case_dice). Subdir tree is appended.",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Checkpoint directory; default: OUT/checkpoints/REGIME/CONFIG/TASK/FM.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--encoder-lr",
        type=float,
        default=None,
        help="full_ft only: Adam lr for a separate encoder param group (head keeps 1e-3). "
        "Default: one group at 1e-3.",
    )
    p.add_argument("--bf16", action="store_true", default=False,
                   help="rejected with an error; training is FP32 only")
    p.add_argument("--log-level", type=str, default="INFO")
    provenance.add_recipe_args(p, primary=False)
    return p.parse_args(argv)


def _config_tag(args: argparse.Namespace) -> str:
    if args.regime == "full_ft":
        return f"ft_e{args.epochs}"
    return "rand_init"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.encoder_lr is not None and (args.regime != "full_ft" or not args.encoder_lr > 0):
        raise SystemExit("--encoder-lr must be positive and requires --regime full_ft")
    with determinism.numerical_failure_record(
        Path(args.out) / args.regime / _config_tag(args) / args.task / args.fm / f"seed_{args.seed}",
        task=args.task, fm=args.fm, regime=args.regime, stage="train_adapted", seed=args.seed,
    ):
        return _run(args)


def _run(args: argparse.Namespace) -> int:
    config_tag = _config_tag(args)
    out_root = Path(args.out)
    out_dir = out_root / args.regime / config_tag / args.task / args.fm
    out_path = out_dir / f"seed_{args.seed}.json"

    ckpt_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else out_root / "checkpoints" / args.regime / config_tag / args.task / args.fm
    )
    ckpt_path = ckpt_dir / f"seed_{args.seed}.pt"

    for path in (out_path, ckpt_path):
        if path.exists():
            raise FileExistsError(f"preserving existing artifact: {path}")

    inputs = provenance.read_input_manifest(args.input_manifest, args.task)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    if args.bf16:
        raise ValueError("training requires FP32; --bf16 is not supported")
    determinism.set_seed(args.seed)

    if args.data_root:
        os.environ["FMPOOL_DATA_ROOT"] = str(args.data_root)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    try:
        if args.regime == "rand_init":
            encoder, spec = build_random_encoder(args.fm)
        else:
            encoder, spec = encoders.build_encoder(args.fm)
    except determinism.NumericalHealthError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("FM %s unloadable: %s", args.fm, exc)
        print(f"[train_adapted] ERROR: cannot load FM {args.fm!r}: {exc}", file=sys.stderr)
        return 2
    encoder = encoder.float().to(device)
    if args.regime != "rand_init":  # a randomly initialised encoder has no pinned state to verify
        provenance.verify_encoder_pin(encoder, inputs, required=False)

    for p in encoder.parameters():
        p.requires_grad = True
    encoder_train_mode = True

    normalise_fn = make_normaliser(spec)
    try:
        train_ds = fmpool_datasets.build_dataset(
            args.task, split="train", transform=normalise_fn
        )
        test_ds = fmpool_datasets.build_dataset(
            args.task, split="test", transform=normalise_fn
        )
    except (FileNotFoundError, NotImplementedError) as exc:
        logger.error("dataset unavailable: %s", exc)
        print(
            f"[train_adapted] ERROR: dataset {args.task!r} unavailable: {exc}",
            file=sys.stderr,
        )
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
    train_loader = DataLoader(train_ds, sampler=determinism.SeededShuffle(train_ds, args.seed), drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    decoder = LinearSegHead(
        in_dim=spec.feature_dim, num_classes=1, out_size=INPUT_HW
    ).to(device)

    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_trainable += sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    logger.info("regime=%s n_trainable_params=%d", args.regime, n_trainable)

    t0 = time.time()
    _train(
        encoder=encoder,
        decoder=decoder,
        loader=train_loader,
        device=device,
        epochs=args.epochs,
        encoder_train_mode=encoder_train_mode,
        encoder_lr=args.encoder_lr,
    )
    train_elapsed = time.time() - t0

    test_ids, per_case = _evaluate(
        encoder=encoder,
        decoder=decoder,
        loader=test_loader,
        device=device,
    )

    state = {f"encoder.{k}": v.detach().cpu() for k, v in encoder.state_dict().items()}
    state.update(
        {f"decoder.{k}": v.detach().cpu() for k, v in decoder.state_dict().items()}
    )
    _atomic_save(state, ckpt_path)
    ckpt_sha = determinism.sha256_file(ckpt_path)

    payload = {
        "run_identity_v2": provenance.run_identity(args, __file__, inputs=inputs,
            initial_state_sha256=initial_state_sha,
            recipe={"version": "secondary_fp32_v2", "optimizer": "Adam", "lr": LR}
            | ({} if args.encoder_lr is None else {"encoder_lr": args.encoder_lr, "head_lr": LR})),
        "task": args.task,
        "fm": args.fm,
        "seed": int(args.seed),
        "regime": args.regime,
        "epochs": int(args.epochs),
        "encoder_lr": LR if args.encoder_lr is None else args.encoder_lr,
        "head_lr": LR,
        "n_trainable_params": int(n_trainable),
        "n_test": int(len(per_case)),
        "test_ids": list(test_ids),
        "per_case_dice": [float(x) for x in per_case.tolist()],
        "mean_dice": float(np.mean(per_case)) if per_case.size else float("nan"),
        "checkpoint_sha256": ckpt_sha,
        "training_elapsed_s": float(train_elapsed),
    }
    body = json.dumps(payload, indent=2).encode("utf-8")
    _atomic_write_bytes(out_path, body)
    logger.info(
        "wrote %s (n_test=%d mean_dice=%.4f)",
        out_path,
        payload["n_test"],
        payload["mean_dice"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
