from __future__ import annotations

import numpy as np
import pytest

from analysis.rater_sensitivity import (
    _configure_deterministic_inference,
    _foreground_mask,
    _headline,
    _parse_args,
    _validation,
    build_payload,
    functional_floor_filter,
    render_markdown,
    within_cross,
)


def test_inference_matches_saved_head_deterministic_kernel_protocol() -> None:
    class FakeCudnn:
        deterministic = False
        benchmark = True

    class FakeCuda:
        seeded = None

        @staticmethod
        def is_available() -> bool:
            return True

        @classmethod
        def manual_seed_all(cls, seed: int) -> None:
            cls.seeded = seed

    class FakeTorch:
        cuda = FakeCuda
        backends = type("Backends", (), {"cudnn": FakeCudnn, "cuda": type("Cuda", (), {"matmul": type("Matmul", (), {})()})()})()
        seeded = None
        deterministic_call = None

        @classmethod
        def manual_seed(cls, seed: int) -> None:
            cls.seeded = seed

        @classmethod
        def use_deterministic_algorithms(
            cls,
            enabled: bool,
            *,
            warn_only: bool,
        ) -> None:
            cls.deterministic_call = (enabled, warn_only)

    _configure_deterministic_inference(FakeTorch, 42)

    assert FakeTorch.seeded == 42
    assert FakeCuda.seeded == 42
    assert FakeCudnn.deterministic is True
    assert FakeCudnn.benchmark is False
    assert FakeTorch.deterministic_call == (True, False)


def test_functional_floor_is_per_bundle_across_all_seeds() -> None:
    per_case = {
        ("keep", 1): np.asarray([0.20, 0.40]),
        ("keep", 2): np.asarray([0.30, 0.30]),
        ("drop", 1): np.asarray([0.29, 0.29]),
        ("drop", 2): np.asarray([0.29, 0.29]),
    }

    filtered = functional_floor_filter(per_case)

    assert set(filtered) == {("keep", 1), ("keep", 2)}


def test_within_cross_uses_symmetric_seed_pairs_and_dinov2_pair() -> None:
    up = np.asarray([0.0, 1.0, 2.0, 3.0])
    down = up[::-1]
    per_case = {}
    for bundle, values in {
        "dinov2_vitb14": up,
        "dinov2_vits14": up,
        "other": down,
    }.items():
        for seed in (1, 2):
            per_case[(bundle, seed)] = values

    result = within_cross(per_case)

    assert result["n_within_pairs"] == 7
    assert result["n_cross_pairs"] == 8
    assert result["within_rbar"] == pytest.approx(1.0)
    assert result["cross_rbar"] == pytest.approx(-1.0)
    assert result["gap"] == pytest.approx(2.0)


def test_headline_distinguishes_inside_range_from_inflation() -> None:
    inside = {
        **{f"rater_{index}": {"gap": float(index)} for index in range(1, 7)},
        "consensus": {"gap": 3.5},
    }
    inflated = {
        **{f"rater_{index}": {"gap": float(index)} for index in range(1, 7)},
        "consensus": {"gap": 6.5},
    }

    assert _headline(inside)["consensus_inside_single_rater_range"] is True
    assert (
        _headline(inflated)["consensus_inflated_relative_to_all_single_raters"]
        is True
    )
    assert _headline(inflated)["amount_above_single_rater_max"] == pytest.approx(0.5)


def test_task_foreground_rules_preserve_cup_and_support_disc() -> None:
    pytest.importorskip("torch")  # _foreground_mask imports the torch-based RIGA loader
    values = np.asarray([[0, 128, 255]], dtype=np.uint8)
    assert _foreground_mask(values, "riga_cup").tolist() == [[False, True, False]]
    assert _foreground_mask(values, "riga_disc").tolist() == [[False, True, True]]
    with pytest.raises(ValueError, match="palette"):
        _foreground_mask(np.asarray([[127]], dtype=np.uint8), "riga_cup")


def test_validation_accepts_any_bounded_gap() -> None:
    point = {"within_rbar": 0.5, "cross_rbar": -0.5, "gap": 1.0}
    assert _validation(point, "riga_cup")["status"] == "PASS"
    assert _validation({**point, "gap": 3.0}, "riga_cup")["status"] == "FAIL"


def test_unknown_schema_and_inexact_replay_are_rejected() -> None:
    raw = {"task": "riga_cup", "n_members": 44, "n_bundles": 11,
           "n_cases": 95, "n_predictions_formed": 4180,
           "mask_diagnostics": {"n_mask_files": 570},
           "released_consensus_trace_check": {"status": "FAIL", "exact_match": False,
                "max_abs_per_case_dice_difference": 0.0003487}}
    with pytest.raises(ValueError, match="schema"):
        build_payload(raw)
    raw["schema_version"] = "l1d_raw_v2"
    raw["mask_diagnostics"]["foreground_rule"] = "raw rater TIFF value == 128"
    with pytest.raises(RuntimeError, match="replay gate"):
        build_payload(raw)


def test_cli_and_markdown_are_task_aware() -> None:
    args = _parse_args(
        [
            "inference",
            "--task",
            "riga_disc",
            "--cache-root",
            "/cache",
            "--repo-root",
            "/repo",
            "--rater-mask-root",
            "/masks",
            "--output",
            "/tmp/raw.json",
        ]
    )
    row = {
        "n_bundles": 11,
        "n_members": 44,
        "within_rbar": 0.8,
        "cross_rbar": 0.2,
        "gap": 0.6,
        "gap_case_bootstrap_95ci": [0.5, 0.7],
    }
    annotations = {annotation: dict(row) for annotation in [
        *(f"rater_{index}" for index in range(1, 7)),
        "consensus",
    ]}
    section = {
        "headline": {
            "interpretation": "Disc result.",
            "consensus_gap": 0.6,
            "single_rater_gap_range": [0.5, 0.7],
        },
        "annotations": annotations,
    }
    markdown = render_markdown(
        {
            "task": "riga_disc",
            "protocol": {"foreground_rule": "raw rater TIFF value >= 128"},
            "primary_gt_specific_floor": section,
            "fixed_consensus_pool_sensitivity": section,
        }
    )

    assert args.task == "riga_disc"
    assert "RIGA Disc" in markdown
    assert "individual Disc masks" in markdown
    assert "raw rater TIFF value >= 128" in markdown


def test_changed_single_rater_is_rejected_even_when_consensus_matches(tmp_path):
    pytest.importorskip("torch")
    import hashlib
    from PIL import Image
    from analysis.rater_sensitivity import _load_rater_masks
    case = 'Magrabia/set/image1.tif'
    folder = tmp_path/'Magrabia/set'; folder.mkdir(parents=True)
    raw = np.zeros((9,9),dtype=np.uint8);raw[1:8,1:8]=255;raw[3:6,3:6]=128
    expected = {}
    for rater in range(1,7):
        path = folder/f'image1-{rater}.tif';Image.fromarray(raw).save(path)
        expected['RIGA-mask/'+str(path.relative_to(tmp_path))] = hashlib.sha256(path.read_bytes()).hexdigest()
    original, diagnostic = _load_rater_masks([case],tmp_path,(9,9),'riga_cup',expected)
    assert len(diagnostic['mask_sha256']) == 6
    raw[3,3] = 255
    Image.fromarray(raw).save(folder/'image1-1.tif')
    with pytest.raises(ValueError,match='rater.*hash'):
        _load_rater_masks([case],tmp_path,(9,9),'riga_cup',expected)
