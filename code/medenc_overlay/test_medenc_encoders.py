"""CPU checks for the medical-encoder overlay (random weights, no downloads).

Each test writes a random-weight checkpoint in the store layout
``$FMPOOL_MEDENC_MODEL_ROOT/<fm>/<file>``, builds the encoder through the
primary ``build_encoder`` path and checks frozen state, the documented dense
shape for a 2x3x224x224 input, and parity with the upstream forward.

RAD-DINO and BiomedParse need public upstream *source* (no weights):
``MEDENC_TEST_TORCH_HOME`` (contains hub/facebookresearch_dinov2_<rev>) with
``FMPOOL_DINOV2_REVISION``, and ``FMPOOL_BIOMEDPARSE_ROOT`` (BiomedParse-v1
checkout). Tests needing that source skip when it is absent; set both to run all 6.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SPEC = importlib.util.spec_from_file_location("medenc_overlay_encoders", Path(__file__).with_name("encoders.py"))
encoders = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = encoders
_SPEC.loader.exec_module(encoders)

X = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(0))
SHAPES = {"retfound_vitl16": (2, 1024, 14, 14), "phikon": (2, 768, 14, 14),
          "rad_dino": (2, 768, 16, 16), "biomedparse": (2, 1536, 7, 7)}
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
NORMS = {"retfound_vitl16": IMAGENET, "phikon": IMAGENET,
         "rad_dino": ((0.5307,) * 3, (0.2583,) * 3),
         "biomedparse": ((123.675 / 255, 116.280 / 255, 103.530 / 255),
                         (58.395 / 255, 57.120 / 255, 57.375 / 255))}


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("FMPOOL_MEDENC_MODEL_ROOT", str(tmp_path))
    monkeypatch.delenv("RETFOUND_CFP_CHECKPOINT", raising=False)
    return tmp_path


def _target(root: Path, fm: str, name: str) -> Path:
    (root / fm).mkdir(parents=True, exist_ok=True)
    return root / fm / name


def _build_and_check(fm: str) -> torch.Tensor:
    model, spec = encoders.build_encoder(fm)
    assert spec.fm_id == fm
    assert (spec.feature_dim, *spec.feature_hw) == SHAPES[fm][1:]
    assert all(not p.requires_grad for p in model.parameters())
    assert not any(m.training for m in model.modules())
    feats = encoders.extract_features(model, fm, X)
    assert tuple(feats.shape) == SHAPES[fm] and feats.dtype == torch.float32
    assert torch.isfinite(feats).all()
    return feats


def test_specs_and_normalisation_match_adapters():
    for fm, (mean, std) in NORMS.items():
        got_mean, got_std = encoders.get_normalisation(fm)
        torch.testing.assert_close(got_mean, torch.tensor(mean), rtol=0, atol=1e-7)
        torch.testing.assert_close(got_std, torch.tensor(std), rtol=0, atol=1e-7)


def test_default_store_root_and_missing_file(monkeypatch, tmp_path):
    monkeypatch.delenv("FMPOOL_MEDENC_MODEL_ROOT", raising=False)
    assert encoders._medenc_model_root() == Path(
        os.environ.get("FMPOOL_MEDENC_ASSET_ROOT", "medenc_assets")) / "models"
    monkeypatch.setenv("FMPOOL_MEDENC_MODEL_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="phikon/model.safetensors"):
        encoders.build_encoder("phikon")


def test_phikon_hf_to_timm_remap_matches_hf_vit(root):
    from safetensors.torch import save_file
    from transformers import ViTConfig, ViTModel

    # ViT-B/16 hyperparameters of owkin/phikon@057cc029 (the config file itself is not redistributed).
    config = ViTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
                       hidden_act="gelu", hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                       initializer_range=0.02, layer_norm_eps=1e-6, image_size=224, patch_size=16,
                       num_channels=3, qkv_bias=True, encoder_stride=16)
    assert config.layer_norm_eps == 1e-6  # owkin/phikon@057cc029 (eps set by commit 1a3b784e)
    torch.manual_seed(1)
    hf = ViTModel(config).eval()  # pinned store revision's config; pooler kept like owkin/phikon
    save_file({k: v.contiguous() for k, v in hf.state_dict().items()},
              str(_target(root, "phikon", "model.safetensors")))
    feats = _build_and_check("phikon")
    model, _ = encoders.build_encoder("phikon")
    assert {m.eps for m in model.modules() if isinstance(m, torch.nn.LayerNorm)} == {config.layer_norm_eps}
    with torch.no_grad():
        ref = hf(pixel_values=X).last_hidden_state[:, 1:]
    ref = ref.transpose(1, 2).reshape(SHAPES["phikon"])
    # Same eps as the pinned HF config; measured max diff 0.0 on CPU.
    torch.testing.assert_close(feats, ref, rtol=0, atol=1e-6)


def _dinov2_env(monkeypatch):
    torch_home = os.environ.get("MEDENC_TEST_TORCH_HOME")
    revision = os.environ.get("FMPOOL_DINOV2_REVISION", "")
    hub = Path(torch_home or "/missing") / "hub" / f"facebookresearch_dinov2_{revision}"
    if not (hub / "hubconf.py").is_file():
        pytest.skip(f"pinned DINOv2 hub source not available in this environment: {hub} (set MEDENC_TEST_TORCH_HOME and FMPOOL_DINOV2_REVISION)")
    monkeypatch.setenv("TORCH_HOME", torch_home)
    return hub


def test_rad_dino_loads_backbone_compatible_into_pinned_dinov2(root, monkeypatch):
    from safetensors.torch import save_file

    hub = _dinov2_env(monkeypatch)
    torch.manual_seed(2)
    ref = torch.hub.load(str(hub), "dinov2_vitb14", source="local", pretrained=False).eval()
    save_file({k: v.contiguous() for k, v in ref.state_dict().items()},
              str(_target(root, "rad_dino", "backbone_compatible.safetensors")))
    feats = _build_and_check("rad_dino")
    with torch.no_grad():
        want = ref.forward_features(X)["x_norm_patchtokens"]
    torch.testing.assert_close(feats, want.transpose(1, 2).reshape(SHAPES["rad_dino"]),
                               rtol=0, atol=0)


FOCAL_CFG = {  # MODEL.BACKBONE of the store config.yaml (git blob 32a4948b)
    "NAME": "focal", "PRETRAINED": "", "LOAD_PRETRAINED": False,
    "FOCAL": {"PRETRAIN_IMG_SIZE": 224, "PATCH_SIZE": 4, "EMBED_DIM": 192,
              "DEPTHS": [2, 2, 18, 2], "FOCAL_LEVELS": [4, 4, 4, 4],
              "FOCAL_WINDOWS": [3, 3, 3, 3], "DROP_PATH_RATE": 0.3, "MLP_RATIO": 4.0,
              "DROP_RATE": 0.0, "PATCH_NORM": True, "USE_CONV_EMBED": True,
              "SCALING_MODULATOR": True, "USE_CHECKPOINT": False, "USE_POSTLN": True,
              "USE_POSTLN_IN_MODULATION": False, "USE_LAYERSCALE": True,
              "OUT_FEATURES": ["res2", "res3", "res4", "res5"], "OUT_INDICES": [0, 1, 2, 3]}}


def test_biomedparse_focal_res5_from_backbone_prefix(root):
    import yaml

    upstream = Path(os.environ.get("FMPOOL_BIOMEDPARSE_ROOT", "/missing"))
    if not (upstream / "modeling/vision/backbone/focal.py").is_file():
        pytest.skip(f"BiomedParse upstream checkout not available in this environment: {upstream} (set FMPOOL_BIOMEDPARSE_ROOT)")
    config = {"VERBOSE": True, "MODEL": {"BACKBONE": FOCAL_CFG}}
    _target(root, "biomedparse", "config.yaml").write_text(yaml.safe_dump(config))
    encoders._install_biomedparse_import_shims(upstream)
    from modeling.vision.backbone.focal import get_focal_backbone

    torch.manual_seed(3)
    ref = get_focal_backbone(config)
    ref.eval()  # upstream FocalNet.train() returns None, so no chaining
    state = {"backbone." + k: v for k, v in ref.state_dict().items()}
    state["sem_seg_head.pixel_decoder.dummy"] = torch.zeros(1)  # must be ignored
    torch.save(state, _target(root, "biomedparse", "biomedparse_v1.pt"))
    feats = _build_and_check("biomedparse")
    with torch.no_grad():
        torch.testing.assert_close(feats, ref(X)["res5"], rtol=0, atol=0)


def test_retfound_store_fallback_namespace_checkpoint_and_trunk_replay(root, monkeypatch):
    import timm

    real_create = timm.create_model
    # Two blocks keep the file small; width/patching are the real ViT-L/16 ones.
    monkeypatch.setattr(timm, "create_model", lambda *a, **kw: real_create(*a, **{**kw, "depth": 2}))
    torch.manual_seed(4)
    ref = real_create("vit_large_patch16_224", pretrained=False, num_classes=0,
                      global_pool="", depth=2).eval()
    model_state = dict(ref.state_dict())
    model_state["mask_token"] = torch.zeros(1, 1, 1024)
    model_state["decoder_embed.weight"] = torch.zeros(512, 1024)
    torch.save({"model": model_state, "args": argparse.Namespace(model="vit_large")},
               _target(root, "retfound_vitl16", "RETFound_mae_natureCFP.pth"))
    feats = _build_and_check("retfound_vitl16")
    with torch.no_grad():  # timm patch features with the CLS token removed
        want = ref.forward_features(X)[:, 1:]
    torch.testing.assert_close(feats, want.transpose(1, 2).reshape(SHAPES["retfound_vitl16"]),
                               rtol=1e-5, atol=1e-5)
