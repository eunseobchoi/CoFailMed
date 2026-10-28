"""Frozen foundation-model encoder wrappers.

Snapshot of ``code/src/fmpool/encoders.py`` with the medical encoders added, used in place of
that file for the medical-encoder runs. The two files are not kept in sync.

Implements the dense-feature API for the primary, expanded and medical encoder pools. Loaders use public
upstream hubs/checkpoints or documented mirrors where available; several
wrappers deliberately expose local dense-probe features rather than vendor
inference APIs. Checkpoint-gated entries require local credentials or explicit
checkpoint paths:

- DINOv2 ViT-S/B/L/14: ``torch.hub.load('facebookresearch/dinov2', ...)``
- BiomedCLIP, CLIP ViT-L/14, CLIP ViT-B/16, CLIP ViT-B/32: ``open_clip``
- ConvNeXt-T/S/B, EfficientNet-B0, ResNet18/34/50/101: ``torchvision``
- MAE ViT-B/16: ``timm.create_model('vit_base_patch16_224.mae', pretrained=True)``
  (official MAE ImageNet weights, mirrored on the timm hub).
- DeiT-B/16 (non-distilled): ``timm.create_model('deit_base_patch16_224',
  pretrained=True)`` (official ``dl.fbaipublicfiles.com/deit/...`` checkpoint).
- RETFound ViT-L/16 CFP: HuggingFace ``YukunZhou/RETFound_mae_natureCFP``
  (gated). Checkpoint can be supplied via ``RETFOUND_CFP_CHECKPOINT`` env var.

Medical-encoder adapters for the upstream checkpoints listed below. Files are read from
``$FMPOOL_MEDENC_MODEL_ROOT/<fm_id>/<file>`` (default
``$FMPOOL_MEDENC_ASSET_ROOT/models``, asset root default ``medenc_assets``):

- ``retfound_vitl16/RETFound_mae_natureCFP.pth`` (``YukunZhou/RETFound_mae_natureCFP``,
  gated); ``RETFOUND_CFP_CHECKPOINT`` still takes precedence.
- ``phikon/model.safetensors`` (``owkin/phikon`` HF ViTModel, remapped to timm).
- ``rad_dino/backbone_compatible.safetensors`` (``microsoft/rad-dino``), loaded
  strictly into the pinned ``FMPOOL_DINOV2_REVISION`` hub ``dinov2_vitb14``.
- ``biomedparse/biomedparse_v1.pt`` + ``biomedparse/config.yaml``
  (``microsoft/BiomedParse``, gated); FocalNet source from the BiomedParse-v1
  git checkout at ``$FMPOOL_BIOMEDPARSE_ROOT`` (default
  ``$FMPOOL_MEDENC_ASSET_ROOT/upstream/BiomedParse-v1``).

All encoders expose :func:`extract_features` returning ``[B, D, h, w]`` dense
features for a ``[B, 3, 224, 224]`` input. They are frozen (``requires_grad
=False``) and in ``eval`` mode at build time. Device management is the
caller's responsibility.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------

_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

_OPENAI_CLIP_MEAN: tuple[float, float, float] = (
    0.48145466,
    0.4578275,
    0.40821073,
)
_OPENAI_CLIP_STD: tuple[float, float, float] = (
    0.26862954,
    0.26130258,
    0.27577711,
)

# RAD-DINO published processor mean/std. BiomedParse's official
# pixel mean/std [123.675,116.28,103.53]/[58.395,57.12,57.375] equal ImageNet /255.
_RAD_DINO_MEAN: tuple[float, float, float] = (0.5307, 0.5307, 0.5307)
_RAD_DINO_STD: tuple[float, float, float] = (0.2583, 0.2583, 0.2583)

_MEDENC_ASSET_ROOT = os.environ.get("FMPOOL_MEDENC_ASSET_ROOT", "medenc_assets")

# ---------------------------------------------------------------------------
# Spec table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderSpec:
    """Static metadata for a single FM encoder."""

    fm_id: str
    feature_dim: int
    feature_hw: tuple[int, int]
    norm_mean: tuple[float, float, float]
    norm_std: tuple[float, float, float]
    family: str


FM_SPECS: dict[str, EncoderSpec] = {
    "dinov2_vitb14": EncoderSpec(
        fm_id="dinov2_vitb14",
        feature_dim=768,
        feature_hw=(16, 16),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="DINOv2",
    ),
    "dinov2_vitl14": EncoderSpec(
        fm_id="dinov2_vitl14",
        feature_dim=1024,
        feature_hw=(16, 16),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="DINOv2",
    ),
    "dinov2_vits14": EncoderSpec(
        fm_id="dinov2_vits14",
        feature_dim=384,
        feature_hw=(16, 16),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="DINOv2",
    ),
    "biomedclip": EncoderSpec(
        fm_id="biomedclip",
        feature_dim=768,
        feature_hw=(14, 14),
        norm_mean=_OPENAI_CLIP_MEAN,
        norm_std=_OPENAI_CLIP_STD,
        family="BiomedCLIP",
    ),
    "clip_vitl14": EncoderSpec(
        fm_id="clip_vitl14",
        feature_dim=1024,
        feature_hw=(16, 16),
        norm_mean=_OPENAI_CLIP_MEAN,
        norm_std=_OPENAI_CLIP_STD,
        family="CLIP",
    ),
    "convnext_tiny": EncoderSpec(
        fm_id="convnext_tiny",
        feature_dim=768,
        feature_hw=(7, 7),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="ConvNeXt",
    ),
    "convnext_small": EncoderSpec(
        fm_id="convnext_small",
        feature_dim=768,
        feature_hw=(7, 7),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="ConvNeXt",
    ),
    "convnext_base": EncoderSpec(
        fm_id="convnext_base",
        feature_dim=1024,
        feature_hw=(7, 7),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="ConvNeXt",
    ),
    "efficientnet_b0": EncoderSpec(
        fm_id="efficientnet_b0",
        feature_dim=1280,
        feature_hw=(7, 7),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="EfficientNet",
    ),
    "resnet50": EncoderSpec(
        fm_id="resnet50",
        feature_dim=2048,
        feature_hw=(7, 7),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="ResNet",
    ),
    "resnet101": EncoderSpec(
        fm_id="resnet101",
        feature_dim=2048,
        feature_hw=(7, 7),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="ResNet",
    ),
    "resnet18": EncoderSpec(
        fm_id="resnet18",
        feature_dim=512,
        feature_hw=(7, 7),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="ResNet",
    ),
    "resnet34": EncoderSpec(
        fm_id="resnet34",
        feature_dim=512,
        feature_hw=(7, 7),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="ResNet",
    ),
    # 5-ViT same-architecture row -------------------------------------------------
    "mae_vitb16": EncoderSpec(
        fm_id="mae_vitb16",
        feature_dim=768,
        feature_hw=(14, 14),  # 224 / 16 = 14
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="MAE",
    ),
    "deit_vitb16": EncoderSpec(
        fm_id="deit_vitb16",
        feature_dim=768,
        feature_hw=(14, 14),
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="DeiT",
    ),
    "clip_vitb16": EncoderSpec(
        fm_id="clip_vitb16",
        feature_dim=768,
        feature_hw=(14, 14),  # 224 / 16 = 14
        norm_mean=_OPENAI_CLIP_MEAN,
        norm_std=_OPENAI_CLIP_STD,
        family="CLIP",
    ),
    "clip_vitb32": EncoderSpec(
        fm_id="clip_vitb32",
        feature_dim=768,
        feature_hw=(7, 7),  # 224 / 32 = 7
        norm_mean=_OPENAI_CLIP_MEAN,
        norm_std=_OPENAI_CLIP_STD,
        family="CLIP",
    ),
    # dense medical FMs ---------------------------------------------------
    "retfound_vitl16": EncoderSpec(
        fm_id="retfound_vitl16",
        feature_dim=1024,
        feature_hw=(14, 14),
        norm_mean=_IMAGENET_MEAN,  # RETFound MAE uses IMAGENET_DEFAULT_MEAN/STD
        norm_std=_IMAGENET_STD,
        family="RETFound",
    ),
    # Medical-encoder adapters ------------
    "phikon": EncoderSpec(
        fm_id="phikon",
        feature_dim=768,
        feature_hw=(14, 14),  # ViT-B/16 at 224
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="Phikon",
    ),
    "rad_dino": EncoderSpec(
        fm_id="rad_dino",
        feature_dim=768,
        feature_hw=(16, 16),  # ViT-B/14 at 224
        norm_mean=_RAD_DINO_MEAN,
        norm_std=_RAD_DINO_STD,
        family="RAD-DINO",
    ),
    "biomedparse": EncoderSpec(
        fm_id="biomedparse",
        feature_dim=1536,
        feature_hw=(7, 7),  # FocalNet res5, stride 32 at 224
        norm_mean=_IMAGENET_MEAN,
        norm_std=_IMAGENET_STD,
        family="BiomedParse",
    ),
}


# ---------------------------------------------------------------------------
# Thin wrapper modules per family
# ---------------------------------------------------------------------------


class _DINOv2Wrapper(nn.Module):
    """Wrap a DINOv2 ViT so ``forward`` returns ``[B, D, 16, 16]``.

    ``model.forward_features(x)["x_norm_patchtokens"]`` reshape.
    """

    def __init__(self, model: nn.Module, feature_dim: int, grid: tuple[int, int]):
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(x)
        patch = feats["x_norm_patchtokens"]  # [B, N, D]
        b, n, d = patch.shape
        h, w = self.grid
        if n != h * w:
            raise RuntimeError(
                f"DINOv2 patch count mismatch: got N={n}, expected {h}*{w}={h * w}. "
                "Input resolution must be 224 so the 14-patch grid yields 16x16."
            )
        return patch.transpose(1, 2).reshape(b, d, h, w).contiguous()


class _BiomedCLIPWrapper(nn.Module):
    """BiomedCLIP visual trunk (timm ViT-B/16) → ``[B, 768, 14, 14]``.

    ``trunk.forward_features(x)[:, 1:]`` then reshape. The timm trunk
    returns ``[B, 1+N, D]`` with the CLS token at index 0.
    """

    def __init__(self, trunk: nn.Module, feature_dim: int, grid: tuple[int, int]):
        super().__init__()
        self.trunk = trunk
        self.feature_dim = feature_dim
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.trunk.forward_features(x)  # [B, 1+N, D]
        patch = feats[:, 1:]
        b, n, d = patch.shape
        h, w = self.grid
        if n != h * w:
            raise RuntimeError(
                f"BiomedCLIP patch count mismatch: got N={n}, expected {h}*{w}={h * w}."
            )
        return patch.transpose(1, 2).reshape(b, d, h, w).contiguous()


class _CLIPViTLWrapper(nn.Module):
    """CLIP ViT-L/14 visual tower → ``[B, 1024, 16, 16]`` pre-projection.

    ``visual.forward_intermediates(x, indices=[-1],
    output_fmt='NCHW', intermediates_only=True)[0]``. This bypasses CLIP's
    final LayerNorm (``ln_post``), token pooling and projection, delivering the final block's dense tokens in
    NCHW layout. Using ``encode_image`` would pool to ``[B, D]`` and break the
    dense probe.
    """

    def __init__(self, visual: nn.Module, feature_dim: int, grid: tuple[int, int]):
        super().__init__()
        self.visual = visual
        self.feature_dim = feature_dim
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.visual.forward_intermediates(
            x,
            indices=[-1],
            output_fmt="NCHW",
            intermediates_only=True,
        )
        # open_clip>=3 returns ``{"image_intermediates": [tensor, ...]}``;
        # older versions returned a bare list/tuple. Handle both without
        # breaking the documented call signature.
        if isinstance(out, dict):
            intermediates = out.get("image_intermediates")
            if intermediates is None:
                raise RuntimeError(
                    "CLIP ViT-L/14 forward_intermediates returned dict without "
                    f"'image_intermediates'; keys={list(out.keys())}"
                )
        else:
            intermediates = out
        feat = intermediates[-1]  # [B, D, h, w]
        expected = (self.feature_dim, *self.grid)
        if tuple(feat.shape[1:]) != expected:
            raise RuntimeError(
                f"CLIP ViT-L/14 feature shape mismatch: got {tuple(feat.shape)}, "
                f"expected [B, {expected[0]}, {expected[1]}, {expected[2]}]."
            )
        return feat


class _CNNWrapper(nn.Module):
    """Generic torchvision CNN trunk → ``[B, D, 7, 7]`` at 224 input.

    ``nn.Sequential(*list(model.children())[:-2])``. Slicing ``[:-2]``
    drops the final avgpool and classifier for ConvNeXt / EfficientNet / ResNet
    alike, leaving the last conv-stage feature map.
    """

    def __init__(self, trunk: nn.Module, feature_dim: int, grid: tuple[int, int]):
        super().__init__()
        self.trunk = trunk
        self.feature_dim = feature_dim
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(x)
        expected = (self.feature_dim, *self.grid)
        if tuple(feat.shape[1:]) != expected:
            raise RuntimeError(
                f"CNN feature shape mismatch: got {tuple(feat.shape)}, "
                f"expected [B, {expected[0]}, {expected[1]}, {expected[2]}]."
            )
        return feat


class _TimmViTWrapper(nn.Module):
    """Wrap a timm VisionTransformer so ``forward`` returns ``[B, D, h, w]``.

    Used for MAE (``vit_base_patch16_224.mae``) and DeiT
    (``deit_base_patch16_224``). Both models expose
    ``forward_features(x) -> [B, 1+N, D]`` with the CLS token at index 0
    (DeiT-base, non-distilled, has a single CLS token; the distilled variant
    has two extra tokens — we deliberately do NOT use that variant).
    """

    def __init__(self, model: nn.Module, feature_dim: int, grid: tuple[int, int]):
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(x)  # [B, 1+N, D] for ViT-B
        if feats.ndim != 3:
            raise RuntimeError(
                f"timm ViT forward_features returned ndim={feats.ndim}; expected 3"
            )
        h, w = self.grid
        n_expected = h * w
        # Drop the CLS token (index 0). If the model already returns the
        # post-pool [B, D] (num_classes=0 plus pre_logits), fail loudly.
        b, t, d = feats.shape
        if t == n_expected + 1:
            patch = feats[:, 1:]
        elif t == n_expected:
            patch = feats
        else:
            raise RuntimeError(
                f"timm ViT token count mismatch: got T={t}, expected {n_expected}"
                f" or {n_expected + 1}."
            )
        if d != self.feature_dim:
            raise RuntimeError(
                f"timm ViT feature_dim mismatch: got D={d}, expected {self.feature_dim}"
            )
        return patch.transpose(1, 2).reshape(b, d, h, w).contiguous()


class _CLIPViTBWrapper(nn.Module):
    """OpenAI CLIP ViT-B visual tower → dense pre-projection patch features.

    Same pattern as :class:`_CLIPViTLWrapper`. ``forward_intermediates`` with
    ``output_fmt='NCHW'`` and ``intermediates_only=True`` bypasses the CLIP
    final LayerNorm (``ln_post``), token pooling and projection, returning the final block's dense tokens
    in NCHW layout.
    """

    def __init__(self, visual: nn.Module, feature_dim: int, grid: tuple[int, int]):
        super().__init__()
        self.visual = visual
        self.feature_dim = feature_dim
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.visual.forward_intermediates(
            x,
            indices=[-1],
            output_fmt="NCHW",
            intermediates_only=True,
        )
        if isinstance(out, dict):
            intermediates = out.get("image_intermediates")
            if intermediates is None:
                raise RuntimeError(
                    "CLIP ViT-B forward_intermediates returned dict without "
                    f"'image_intermediates'; keys={list(out.keys())}"
                )
        else:
            intermediates = out
        feat = intermediates[-1]
        expected = (self.feature_dim, *self.grid)
        if tuple(feat.shape[1:]) != expected:
            raise RuntimeError(
                f"CLIP ViT-B feature shape mismatch: got {tuple(feat.shape)}, "
                f"expected [B, {expected[0]}, {expected[1]}, {expected[2]}]."
            )
        return feat


class _RETFoundWrapper(nn.Module):
    """RETFound ViT-L/16 → ``[B, 1024, 14, 14]`` dense patch tokens.

    Wraps the timm ViT-L/16 trunk loaded with the official RETFound weights.
    Replays patch_embed → CLS/pos_embed → pos_drop → blocks → norm, then
    drops the CLS token to retain dense features.
    """

    def __init__(self, model: nn.Module, feature_dim: int, grid: tuple[int, int]):
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        # RETFound's models_vit defines forward_features that returns the CLS
        # token only when global_pool=False. To get patch tokens we replay the
        # trunk manually.
        b = x.shape[0]
        x = m.patch_embed(x)
        cls_tokens = m.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + m.pos_embed
        x = m.pos_drop(x)
        for blk in m.blocks:
            x = blk(x)
        x = m.norm(x)
        # Drop CLS token at position 0 → [B, N, D]
        patch = x[:, 1:]
        bn, n, d = patch.shape
        h, w = self.grid
        if n != h * w:
            raise RuntimeError(
                f"RETFound patch count mismatch: got N={n}, expected {h * w}."
            )
        if d != self.feature_dim:
            raise RuntimeError(
                f"RETFound feature_dim mismatch: got D={d}, expected {self.feature_dim}"
            )
        return patch.transpose(1, 2).reshape(bn, d, h, w).contiguous()


class _BiomedParseWrapper(nn.Module):
    """BiomedParse FocalNet backbone → final stage ``res5`` ``[B, 1536, 7, 7]``."""

    def __init__(self, backbone: nn.Module, feature_dim: int, grid: tuple[int, int]):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)["res5"]
        expected = (self.feature_dim, *self.grid)
        if tuple(feat.shape[1:]) != expected:
            raise RuntimeError(
                f"BiomedParse res5 shape mismatch: got {tuple(feat.shape)}, "
                f"expected [B, {expected[0]}, {expected[1]}, {expected[2]}]."
            )
        return feat


# ---------------------------------------------------------------------------
# Per-FM loaders: public upstream/hub checkpoints plus documented local
# dense-probe wrappers.


def _load_dinov2(fm_id: str, *, pretrained: bool = True) -> nn.Module:
    # torch.hub.load yields the DINOv2 ViT with ``forward_features`` returning
    # the dict that contains ``x_norm_patchtokens``.
    revision = os.environ.get("FMPOOL_DINOV2_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("FMPOOL_DINOV2_REVISION must pin a full 40-character commit")
    # skip_validation: torch.hub otherwise rejects a pinned commit that is no longer a
    # branch or tag head; the archive of the pinned SHA is still downloaded.
    model = torch.hub.load(f"facebookresearch/dinov2:{revision}", fm_id, pretrained=pretrained,
                           skip_validation=True)
    spec = FM_SPECS[fm_id]
    return _DINOv2Wrapper(model, spec.feature_dim, spec.feature_hw)


def _load_biomedclip() -> nn.Module:
    import open_clip

    model, _ = open_clip.create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    # open_clip CustomTextCLIP exposes the timm trunk via ``visual.trunk``.
    visual = model.visual
    trunk = visual.trunk if hasattr(visual, "trunk") else visual
    spec = FM_SPECS["biomedclip"]
    return _BiomedCLIPWrapper(trunk, spec.feature_dim, spec.feature_hw)


def _load_clip_vitl14() -> nn.Module:
    """Canonical OpenAI QuickGELU visual tower."""
    import open_clip

    out = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="openai", precision="fp32")
    model = out[0]
    spec = FM_SPECS["clip_vitl14"]
    return _CLIPViTLWrapper(model.visual, spec.feature_dim, spec.feature_hw)


def _load_torchvision_cnn(fm_id: str) -> nn.Module:
    import torchvision.models as tvm

    if fm_id == "convnext_tiny":
        model = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    elif fm_id == "convnext_small":
        model = tvm.convnext_small(weights=tvm.ConvNeXt_Small_Weights.IMAGENET1K_V1)
    elif fm_id == "convnext_base":
        model = tvm.convnext_base(weights=tvm.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    elif fm_id == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif fm_id == "resnet50":
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    elif fm_id == "resnet101":
        model = tvm.resnet101(weights=tvm.ResNet101_Weights.IMAGENET1K_V1)
    elif fm_id == "resnet18":
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    elif fm_id == "resnet34":
        model = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown torchvision FM id: {fm_id!r}")

    trunk = nn.Sequential(*list(model.children())[:-2])
    spec = FM_SPECS[fm_id]
    return _CNNWrapper(trunk, spec.feature_dim, spec.feature_hw)


def _load_mae_vitb16() -> nn.Module:
    """Load MAE ViT-B/16 ImageNet-pretrained weights via timm.

    Source: ``timm.create_model('vit_base_patch16_224.mae', pretrained=True)``.
    The timm hub mirrors the official MAE checkpoint
    (``https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth``,
    sha256 unpublished by upstream). Setting ``num_classes=0`` removes the
    classifier head; we still call ``forward_features`` to obtain dense tokens.
    """

    import timm

    model = timm.create_model(
        "vit_base_patch16_224.mae", pretrained=True, num_classes=0
    )
    spec = FM_SPECS["mae_vitb16"]
    return _TimmViTWrapper(model, spec.feature_dim, spec.feature_hw)


def _load_deit_vitb16() -> nn.Module:
    """Load DeiT-B/16 (non-distilled) via timm.

    Source: ``timm.create_model('deit_base_patch16_224', pretrained=True)``,
    backed by the official checkpoint
    ``https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth``
    (filename hash ``b5f2ef4d`` is part of the URL, not a checksum). This is
    the NON-distilled DeiT-B/16.
    """

    import timm

    model = timm.create_model(
        "deit_base_patch16_224", pretrained=True, num_classes=0
    )
    spec = FM_SPECS["deit_vitb16"]
    return _TimmViTWrapper(model, spec.feature_dim, spec.feature_hw)


def _load_clip_vitb16() -> nn.Module:
    """Load OpenAI CLIP ViT-B/16 visual tower via open_clip.

    Source: ``open_clip.create_model_and_transforms('ViT-B-16-quickgelu',
    pretrained='openai')``. The ``-quickgelu`` suffix matches OpenAI's
    original activation (per open_clip README: "OpenAI pretrained weights
    will always default to QuickGELU"). The visual tower's pre-projection
    feature dim is 768 and the patch-token grid at 224 input is 14x14.
    """

    import open_clip

    out = open_clip.create_model_and_transforms(
        "ViT-B-16-quickgelu", pretrained="openai"
    )
    model = out[0]
    spec = FM_SPECS["clip_vitb16"]
    return _CLIPViTBWrapper(model.visual, spec.feature_dim, spec.feature_hw)


def _load_clip_vitb32() -> nn.Module:
    """Load OpenAI CLIP ViT-B/32 visual tower via open_clip.

    Source: ``open_clip.create_model_and_transforms('ViT-B-32-quickgelu',
    pretrained='openai')``. We use the QuickGELU model definition for the
    original OpenAI checkpoint and expose the pre-projection 7x7 patch map at
    224 input.
    """

    import open_clip

    out = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai"
    )
    model = out[0]
    spec = FM_SPECS["clip_vitb32"]
    return _CLIPViTBWrapper(model.visual, spec.feature_dim, spec.feature_hw)


def _load_retfound_vitl16() -> nn.Module:
    """Load RETFound ViT-L/16 (CFP, gated HuggingFace checkpoint).

    Source: ``YukunZhou/RETFound_mae_natureCFP`` on HuggingFace (gated). To
    use this loader you must either:

    1. Set ``RETFOUND_CFP_CHECKPOINT`` to a local ``.pth`` checkpoint path, OR
       place it at ``$FMPOOL_MEDENC_MODEL_ROOT/retfound_vitl16/RETFound_mae_natureCFP.pth``, OR
    2. Be authenticated to HF (``huggingface-cli login``) so that
       ``hf_hub_download`` can fetch ``RETFound_mae_natureCFP.pth``.

    Architecture matches the official ``models_vit.RETFound_mae``:
    embed_dim=1024, depth=24, num_heads=16, patch_size=16, mlp_ratio=4,
    qkv_bias=True. We build the trunk locally (using timm's
    VisionTransformer) and load the official state_dict with strict=False
    (the checkpoint contains an MAE decoder which we drop).
    """

    import timm

    # Build the encoder skeleton with the exact RETFound architecture.
    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=0,
        global_pool="",  # keep CLS+patch tokens
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
    )
    # Compatibility: RETFound's official trunk lives in ``models_vit`` and uses
    # ``pos_drop`` + 24 ``blocks`` + ``norm`` layer names — these match timm's
    # VisionTransformer exactly, so the wrapper's forward replay is valid.

    ckpt_path = os.environ.get("RETFOUND_CFP_CHECKPOINT")
    store_copy = _medenc_model_root() / "retfound_vitl16" / "RETFound_mae_natureCFP.pth"
    if ckpt_path is None and store_copy.is_file():
        ckpt_path = str(store_copy)
    if ckpt_path is None:
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(
            repo_id="YukunZhou/RETFound_mae_natureCFP",
            filename="RETFound_mae_natureCFP.pth",
        )

    # The official checkpoint pickles its training args (argparse.Namespace);
    # torch>=2.6 weights_only loading rejects it unless allow-listed.
    torch.serialization.add_safe_globals([argparse.Namespace])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    # The MAE checkpoint contains 'decoder_*' and 'mask_token' keys we don't
    # need for the encoder. strict=False allows those extras, but missing
    # encoder keys are treated as a hard preflight failure.
    missing, unexpected = model.load_state_dict(state, strict=False)
    encoder_missing = [k for k in missing if not k.startswith("head")]
    if encoder_missing:
        raise RuntimeError(
            "RETFound checkpoint is missing encoder keys "
            f"(n={len(encoder_missing)}, sample={encoder_missing[:5]}). "
            "Refusing to run the encoder-only probe with a partial trunk."
        )
    ignored_unexpected = [
        k for k in unexpected
        if k.startswith("decoder_") or k in {"mask_token"} or k.startswith("head")
    ]
    unexpected_encoder = sorted(set(unexpected) - set(ignored_unexpected))
    if unexpected_encoder:
        logger.info(
            "RETFound checkpoint had %d non-loaded non-decoder keys; sample=%s",
            len(unexpected_encoder),
            unexpected_encoder[:5],
        )
    spec = FM_SPECS["retfound_vitl16"]
    return _RETFoundWrapper(model, spec.feature_dim, spec.feature_hw)


def _medenc_model_root() -> Path:
    return Path(os.environ.get("FMPOOL_MEDENC_MODEL_ROOT", f"{_MEDENC_ASSET_ROOT}/models"))


def _medenc_file(fm_id: str, filename: str) -> Path:
    path = _medenc_model_root() / fm_id / filename
    if not path.is_file():
        raise FileNotFoundError(f"medical encoder checkpoint is missing: {path}")
    return path


def _phikon_timm_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """HF ``ViTModel`` keys → timm ``vit_base_patch16_224``."""
    prefix = "vit." if "vit.embeddings.cls_token" in state else ""
    mapped = {
        "cls_token": state[f"{prefix}embeddings.cls_token"],
        "pos_embed": state[f"{prefix}embeddings.position_embeddings"],
        "patch_embed.proj.weight": state[f"{prefix}embeddings.patch_embeddings.projection.weight"],
        "patch_embed.proj.bias": state[f"{prefix}embeddings.patch_embeddings.projection.bias"],
        "norm.weight": state[f"{prefix}layernorm.weight"],
        "norm.bias": state[f"{prefix}layernorm.bias"],
    }
    for index in range(12):
        source = f"{prefix}encoder.layer.{index}"
        target = f"blocks.{index}"
        attention = f"{source}.attention.attention"
        mapped[f"{target}.norm1.weight"] = state[f"{source}.layernorm_before.weight"]
        mapped[f"{target}.norm1.bias"] = state[f"{source}.layernorm_before.bias"]
        mapped[f"{target}.norm2.weight"] = state[f"{source}.layernorm_after.weight"]
        mapped[f"{target}.norm2.bias"] = state[f"{source}.layernorm_after.bias"]
        mapped[f"{target}.attn.qkv.weight"] = torch.cat(
            [state[f"{attention}.{name}.weight"] for name in ("query", "key", "value")])
        mapped[f"{target}.attn.qkv.bias"] = torch.cat(
            [state[f"{attention}.{name}.bias"] for name in ("query", "key", "value")])
        mapped[f"{target}.attn.proj.weight"] = state[f"{source}.attention.output.dense.weight"]
        mapped[f"{target}.attn.proj.bias"] = state[f"{source}.attention.output.dense.bias"]
        mapped[f"{target}.mlp.fc1.weight"] = state[f"{source}.intermediate.dense.weight"]
        mapped[f"{target}.mlp.fc1.bias"] = state[f"{source}.intermediate.dense.bias"]
        mapped[f"{target}.mlp.fc2.weight"] = state[f"{source}.output.dense.weight"]
        mapped[f"{target}.mlp.fc2.bias"] = state[f"{source}.output.dense.bias"]
    return mapped


def _load_phikon() -> nn.Module:
    """Phikon (iBOT ViT-B/16 histopathology) via timm; patch tokens → 14x14.

    LayerNorm eps=1e-6 = timm default = official owkin/phikon config.json
    ``layer_norm_eps`` since HF commit 1a3b784e (2023-10-31), incl. store revision
    057cc029.
    """
    import timm
    from safetensors.torch import load_file

    model = timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="")
    state = load_file(str(_medenc_file("phikon", "model.safetensors")))
    model.load_state_dict(_phikon_timm_state(state), strict=True)
    spec = FM_SPECS["phikon"]
    return _TimmViTWrapper(model, spec.feature_dim, spec.feature_hw)


def _load_rad_dino() -> nn.Module:
    """RAD-DINO: official DINOv2 ViT-B/14 code at the pinned revision, strict load."""
    from safetensors.torch import load_file

    revision = os.environ.get("FMPOOL_DINOV2_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("FMPOOL_DINOV2_REVISION must pin a full 40-character commit")
    model = torch.hub.load(
        f"facebookresearch/dinov2:{revision}", "dinov2_vitb14", pretrained=False,
        skip_validation=True)  # pinned SHA need not be a branch head
    state = load_file(str(_medenc_file("rad_dino", "backbone_compatible.safetensors")))
    model.load_state_dict(state, strict=True)
    spec = FM_SPECS["rad_dino"]
    return _DINOv2Wrapper(model, spec.feature_dim, spec.feature_hw)


def _install_biomedparse_import_shims(upstream: Path) -> None:
    """Import the official FocalNet file without XDecoder/detectron2."""
    for name, path in (
        ("modeling", upstream / "modeling"),
        ("modeling.vision", upstream / "modeling" / "vision"),
        ("modeling.vision.backbone", upstream / "modeling" / "vision" / "backbone"),
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
    try:
        import detectron2  # noqa: F401
    except ModuleNotFoundError:
        detectron2 = types.ModuleType("detectron2")
        modeling = types.ModuleType("detectron2.modeling")
        utils = types.ModuleType("detectron2.utils")
        file_io = types.ModuleType("detectron2.utils.file_io")

        class ShapeSpec:
            def __init__(self, **kwargs) -> None:
                self.__dict__.update(kwargs)

        class Registry:
            def register(self, value=None):
                return (lambda item: item) if value is None else value

        class PathManager:
            open = staticmethod(open)

        modeling.BACKBONE_REGISTRY = Registry()
        modeling.Backbone = nn.Module
        modeling.ShapeSpec = ShapeSpec
        file_io.PathManager = PathManager
        detectron2.modeling = modeling
        detectron2.utils = utils
        utils.file_io = file_io
        sys.modules.update({
            "detectron2": detectron2,
            "detectron2.modeling": modeling,
            "detectron2.utils": utils,
            "detectron2.utils.file_io": file_io,
        })


def _load_biomedparse() -> nn.Module:
    """BiomedParse v1 FocalNet backbone (``backbone.*`` weights) → res5."""
    import yaml

    upstream = Path(os.environ.get(
        "FMPOOL_BIOMEDPARSE_ROOT", f"{_MEDENC_ASSET_ROOT}/upstream/BiomedParse-v1"))
    _install_biomedparse_import_shims(upstream)
    from modeling.vision.backbone.focal import get_focal_backbone

    config = yaml.safe_load(
        _medenc_file("biomedparse", "config.yaml").read_text(encoding="utf-8"))
    backbone = get_focal_backbone(config)
    state = torch.load(_medenc_file("biomedparse", "biomedparse_v1.pt"),
                       map_location="cpu", weights_only=True, mmap=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in state.items() if key.startswith("backbone.")
    }
    if not backbone_state:
        raise RuntimeError("BiomedParse checkpoint contains no backbone.* weights")
    backbone.load_state_dict(backbone_state, strict=True, assign=True)
    spec = FM_SPECS["biomedparse"]
    return _BiomedParseWrapper(backbone, spec.feature_dim, spec.feature_hw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_encoder(fm_id: str) -> tuple[nn.Module, EncoderSpec]:
    """Load a frozen FM encoder from its documented hub/checkpoint path.

    Parameters
    ----------
    fm_id:
        One of the ``FM_SPECS`` identifiers.

    Returns
    -------
    (model, spec):
        ``model`` is already in ``eval()`` mode with all parameters frozen.
        The caller is responsible for moving it to a device.
    """

    if fm_id not in FM_SPECS:
        raise KeyError(
            f"Unknown fm_id {fm_id!r}. Supported: {sorted(FM_SPECS.keys())}"
        )
    spec = FM_SPECS[fm_id]

    if fm_id.startswith("dinov2_"):
        model = _load_dinov2(fm_id)
    elif fm_id == "biomedclip":
        model = _load_biomedclip()
    elif fm_id == "clip_vitl14":
        model = _load_clip_vitl14()
    elif fm_id == "mae_vitb16":
        model = _load_mae_vitb16()
    elif fm_id == "deit_vitb16":
        model = _load_deit_vitb16()
    elif fm_id == "clip_vitb16":
        model = _load_clip_vitb16()
    elif fm_id == "clip_vitb32":
        model = _load_clip_vitb32()
    elif fm_id == "retfound_vitl16":
        model = _load_retfound_vitl16()
    elif fm_id == "phikon":
        model = _load_phikon()
    elif fm_id == "rad_dino":
        model = _load_rad_dino()
    elif fm_id == "biomedparse":
        model = _load_biomedparse()
    else:
        model = _load_torchvision_cnn(fm_id)

    model.float()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    logger.info("Built encoder fm_id=%s feature_dim=%d", fm_id, spec.feature_dim)
    return model, spec


@torch.no_grad()
def extract_features(
    model: nn.Module, fm_id: str, x: torch.Tensor
) -> torch.Tensor:
    """Run the encoder on ``x`` and return ``[B, D, h, w]`` dense features.

    ``x`` must already be normalised with the per-FM mean/std returned by
    :func:`get_normalisation`. No pooling is applied.
    """

    if fm_id not in FM_SPECS:
        raise KeyError(f"Unknown fm_id {fm_id!r}")
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(
            f"Expected input of shape [B, 3, H, W]; got {tuple(x.shape)}"
        )

    spec = FM_SPECS[fm_id]
    feat = model(x)

    expected_shape = (x.shape[0], spec.feature_dim, *spec.feature_hw)
    if tuple(feat.shape) != expected_shape:
        raise RuntimeError(
            f"{fm_id}: extract_features produced {tuple(feat.shape)}, "
            f"expected {expected_shape}"
        )
    return feat


def get_normalisation(fm_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(mean, std)`` as 1-D tensors of length 3 for ``fm_id``."""

    if fm_id not in FM_SPECS:
        raise KeyError(f"Unknown fm_id {fm_id!r}")
    spec = FM_SPECS[fm_id]
    mean = torch.tensor(spec.norm_mean, dtype=torch.float32)
    std = torch.tensor(spec.norm_std, dtype=torch.float32)
    return mean, std


__all__ = [
    "EncoderSpec",
    "FM_SPECS",
    "build_encoder",
    "extract_features",
    "get_normalisation",
]
