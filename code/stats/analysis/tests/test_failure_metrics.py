from __future__ import annotations

import numpy as np
import pytest

from analysis.failure_metrics import (
    cohen_kappa,
    disagreement_rate,
    double_fault_rate,
    empirical_threshold,
    metric_summary,
    q_statistic,
)


IDENTICAL = np.asarray([False, False, True, True])
OPPOSITE = np.asarray([True, True, False, False])
INDEPENDENT_LEFT = np.asarray([False, False, True, True])
INDEPENDENT_RIGHT = np.asarray([False, True, False, True])


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (cohen_kappa, 1.0),
        (q_statistic, 1.0),
        (disagreement_rate, 0.0),
        (double_fault_rate, 0.5),
    ],
)
def test_metrics_for_identical_success_events(metric, expected: float) -> None:
    assert metric(IDENTICAL, IDENTICAL) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (cohen_kappa, -1.0),
        (q_statistic, -1.0),
        (disagreement_rate, 1.0),
        (double_fault_rate, 0.0),
    ],
)
def test_metrics_for_opposite_success_events(metric, expected: float) -> None:
    assert metric(IDENTICAL, OPPOSITE) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (cohen_kappa, 0.0),
        (q_statistic, 0.0),
        (disagreement_rate, 0.5),
        (double_fault_rate, 0.25),
    ],
)
def test_metrics_for_independent_balanced_events(metric, expected: float) -> None:
    assert metric(INDEPENDENT_LEFT, INDEPENDENT_RIGHT) == pytest.approx(expected)


def test_undefined_adjusted_metrics_return_none() -> None:
    all_success = np.ones(4, dtype=bool)
    assert cohen_kappa(all_success, all_success, kappa_never_fail="exclude") is None
    assert q_statistic(all_success, all_success) is None


def test_never_failing_pair_kappa_follows_official_convention() -> None:
    # wichmann-lab/error-consistency@eb53bdb data-analysis-helper.R:242-248: kappa := 1 when c_obs == 1.
    all_success = np.ones(4, dtype=bool)
    assert cohen_kappa(all_success, all_success) == 1.0
    assert cohen_kappa(all_success, all_success, kappa_never_fail="official") == 1.0
    with pytest.raises(ValueError):
        cohen_kappa(all_success, all_success, kappa_never_fail="drop")


def test_stratum_mean_scores_or_drops_never_failing_pair() -> None:
    # Members a0/a1 (same bundle) never fail; b0 fails on case 0 (cross pairs a-b: kappa 0).
    keys = [("a", 0), ("a", 1), ("b", 0)]
    success = np.asarray([[True, True, True], [True, True, True], [False, True, True]])
    official = metric_summary(keys, success)["kappa"]
    exclude = metric_summary(keys, success, kappa_never_fail="exclude")["kappa"]
    assert (official["same"], official["n_valid_same_pairs"]) == (1.0, 1)
    assert (official["cross"], official["n_valid_cross_pairs"]) == (0.0, 2)
    assert np.isnan(exclude["same"]) and exclude["n_valid_same_pairs"] == 0
    assert (exclude["cross"], exclude["n_valid_cross_pairs"]) == (0.0, 2)


def test_empirical_threshold_is_poolwide_percentile() -> None:
    values = np.asarray([[0.0, 0.2], [0.6, 1.0]])
    assert empirical_threshold(values, 50.0) == pytest.approx(0.4)
