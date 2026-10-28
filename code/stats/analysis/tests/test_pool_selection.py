from __future__ import annotations

import numpy as np
import pytest

from analysis.pool_core import (
    architecture_family,
    calibration_summary,
    evaluation_vectors,
    split_units,
    top_quality_pool,
)
from analysis.pool_evaluation import point
from analysis.pool_selection import RULE_DEFINITION
from analysis.pool_repeated import (
    bootstrap_mean,
    empirical_p33,
)


def test_subject_split_has_no_patient_leakage_and_is_reproducible() -> None:
    case_ids = (
        "patient001_frame01_slice01",
        "patient001_frame01_slice02",
        "patient002_frame01_slice01",
        "patient002_frame01_slice02",
        "patient003_frame01_slice01",
        "patient003_frame01_slice02",
        "patient004_frame01_slice01",
        "patient004_frame01_slice02",
    )

    first = split_units("acdc_lv", case_ids, seed=0)
    second = split_units("acdc_lv", case_ids, seed=0)

    assert first == second
    assert set(first.calibration_units).isdisjoint(first.evaluation_units)
    assert set(first.calibration_columns).isdisjoint(first.evaluation_columns)
    assert len(first.calibration_units) == len(first.evaluation_units) == 2


def test_quality_gate_uses_best_models_standard_error() -> None:
    matrix = np.asarray(
        [
            [0.8, 0.9, 0.7, 0.8],
            [0.76, 0.76, 0.76, 0.76],
            [0.5, 0.5, 0.5, 0.5],
        ]
    )

    summary = calibration_summary(("a", "b", "c"), matrix, floor=0.30)

    expected_se = np.std(matrix[0], ddof=1) / np.sqrt(matrix.shape[1])
    assert summary.best_model == "a"
    assert summary.quality_cutoff == pytest.approx(0.8 - expected_se)
    assert summary.quality_eligible == ("a", "b")
    assert summary.functional_candidates == ("a", "b", "c")


def test_bundle_gate_passes_all_seed_models_from_eligible_bundle() -> None:
    keys = ("strong/seed_1", "strong/seed_2", "weak/seed_1", "weak/seed_2")
    matrix = np.asarray(
        [
            [0.8, 0.9, 0.7, 0.8],
            [0.8, 0.9, 0.7, 0.8],
            [0.4, 0.5, 0.3, 0.4],
            [0.4, 0.5, 0.3, 0.4],
        ]
    )

    summary = calibration_summary(keys, matrix, floor=0.30)

    assert summary.quality_eligible_bundles == ("strong",)
    assert summary.quality_eligible == ("strong/seed_1", "strong/seed_2")


def test_top_quality_ties_are_lexicographic() -> None:
    means = {"z": 0.8, "a": 0.8, "m": 0.7}
    assert top_quality_pool(("z", "a", "m"), means, size=2) == ("a", "z")


@pytest.mark.parametrize(
    ("bundle", "family"),
    [
        ("resnet18", "resnet"),
        ("resnet50", "resnet"),
        ("clip_vitb16", "clip"),
        ("biomedclip", "biomedclip"),
        ("dinov2_vits14", "dinov2"),
        ("convnext_tiny", "convnext"),
    ],
)
def test_architecture_family(bundle: str, family: str) -> None:
    assert architecture_family(f"{bundle}/seed_42") == family


def test_evaluation_vectors_report_all_fail_recovery_and_marginal_dice() -> None:
    keys = ("anchor", "helper")
    matrix = np.asarray(
        [
            [0.4, 0.4, 0.9, 0.9],
            [0.9, 0.4, 0.9, 0.4],
        ]
    )

    vectors = evaluation_vectors(
        keys,
        matrix,
        pools=(("anchor", "helper"),),
        anchor="anchor",
        threshold=0.5,
    )

    assert vectors.all_fail.tolist() == [0.0, 1.0, 0.0, 0.0]
    assert vectors.anchor_fail.tolist() == [True, True, False, False]
    assert vectors.recovered.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert vectors.marginal_dice.tolist() == pytest.approx([0.65, 0.4, 0.9, 0.65])


def test_conditional_recovery_is_undefined_without_anchor_failures() -> None:
    vectors = evaluation_vectors(
        ("anchor", "helper"),
        np.asarray([[0.9, 0.8], [0.7, 0.6]]),
        pools=(("anchor", "helper"),),
        anchor="anchor",
        threshold=0.5,
    )

    assert point(vectors)["conditional_recovery"] is None


def test_rule_text_records_every_prospective_constraint() -> None:
    for phrase in (
        "calibration split",
        "encoder family",
        "Dice >= 0.30",
        "P33",
        "M=4",
        "evaluation",
    ):
        assert phrase in RULE_DEFINITION


def test_empirical_p33_uses_only_anchor_calibration_cases() -> None:
    keys = ("anchor", "other", "excluded")
    calibration_cases = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.0, 0.0, 0.0],
        ]
    )

    threshold = empirical_p33(
        keys,
        calibration_cases,
        anchor="anchor",
    )

    assert threshold == pytest.approx(np.percentile(calibration_cases[0], 33.0))


def test_bootstrap_mean_drops_only_undefined_splits() -> None:
    summary = bootstrap_mean(
        np.asarray([0.2, np.nan, 0.4]),
        draws=500,
        seed=3,
    )

    assert summary["value"] == pytest.approx(0.3)
    assert summary["n_usable_splits"] == 2


def test_tasks_restriction_keeps_default_order_seeds_and_labels_scope(monkeypatch, tmp_path) -> None:
    import analysis.pool_task as pool_task
    from analysis.pool_config import TASKS, scope_tasks
    from analysis.pool_selection import main
    from analysis.pool_reporting import _tasks

    assert scope_tasks(None) == (TASKS, None) and scope_tasks([]) == (TASKS, None)
    tasks, scope = scope_tasks(["riga_cup", "kvasir"])
    assert tasks == ("kvasir", "riga_cup") and scope["omitted_tasks"][0] == "acdc_lv"
    assert scope["label"].startswith("RESTRICTED SCOPE: 2 of 7")
    with pytest.raises(ValueError):
        scope_tasks(["msd_spleen"])

    calls = []
    monkeypatch.setattr(pool_task, "run_task", lambda root, task, index, *a: calls.append((task, index)) or {})
    restricted = pool_task.run_analysis(tmp_path, 2, tasks)
    assert calls == [("kvasir", 0), ("riga_cup", 3)]          # fixed per-task seed index
    assert restricted["protocol"]["tasks"] == tasks
    calls.clear()
    full = pool_task.run_analysis(tmp_path, 2)
    assert [t for t, _ in calls] == list(TASKS) and full["protocol"]["tasks"] == TASKS
    assert "scope" not in full
    assert _tasks({"tasks": {"riga_cup": 0, "kvasir": 0}}) == ["kvasir", "riga_cup"]
    with pytest.raises(SystemExit):   # a restricted run may not overwrite the full seven-task outputs
        main(["--tasks", "kvasir", "--results-root", str(tmp_path)])




def test_family_selection_ignores_evaluation_values_and_keeps_distinct_families():
    from analysis.pool_task import _run_split

    keys = ("resnet18/seed_42", "resnet50/seed_42", "clip_vitb16/seed_42",
            "dinov2_vitb14/seed_42", "biomedclip/seed_42", "deit_vitb16/seed_42")
    cases = tuple(f"case_{i:02d}" for i in range(40))
    matrix = np.random.default_rng(21).uniform(.4, .95, (len(keys), len(cases)))
    _, first, _ = _run_split("kvasir", 0, 3, keys, matrix, cases)
    changed = matrix.copy()
    changed[:, split_units("kvasir", cases, 3).evaluation_columns] = 0
    _, second, _ = _run_split("kvasir", 0, 3, keys, changed, cases)
    for field in ("selections", "tau", "anchor", "functional_candidates", "quality_eligible"):
        assert first[field] == second[field]
    assert set(first["selections"]) == {"architecture_diversity"}
    pool = first["selections"]["architecture_diversity"]["pool"]
    assert len({architecture_family(member) for member in pool}) == 4


def test_family_metric_bootstrap_keeps_published_random_stream():
    from analysis.common import BOOTSTRAP_B
    from analysis.pool_config import METRIC_NAMES
    from analysis.pool_task import _aggregate_threshold

    rows = [{"rules": {"architecture_diversity": dict.fromkeys(METRIC_NAMES, value)},
             "all_fail_case_events_by_rule": {"architecture_diversity": 2}}
            for value in [.1, .3, .7, .2]]
    actual = _aggregate_threshold(rows, 100_000)["rules"]["architecture_diversity"]
    for index, metric in enumerate(METRIC_NAMES):
        assert actual[metric] == bootstrap_mean(np.array([.1, .3, .7, .2]),
                                               BOOTSTRAP_B, 100_000 + 4 * 101 + index)
