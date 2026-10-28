from __future__ import annotations

import numpy as np
import pytest

from analysis.trace_validation import (
    PEARSON_STD_FLOOR,
    Trace,
    align_pair,
    apply_bundle_floor,
    pearson,
    pearson_input_has_variance,
    within_cross,
)


def _trace(
    bundle: str,
    seed: int,
    values: list[float],
    ids: tuple[str, ...] = ("a", "b", "c", "d"),
) -> Trace:
    return Trace(
        task="synthetic",
        bundle=bundle,
        seed=seed,
        test_ids=ids,
        dice=np.asarray(values, dtype=float),
        source=f"{bundle}/seed_{seed}.json",
    )


@pytest.mark.parametrize(
    ("right", "expected"),
    [
        ([0.0, 1.0, 2.0, 3.0], 1.0),
        ([3.0, 2.0, 1.0, 0.0], -1.0),
        ([1.0, -1.0, -1.0, 1.0], 0.0),
    ],
)
def test_pearson_known_values(right: list[float], expected: float) -> None:
    left = np.asarray([0.0, 1.0, 2.0, 3.0])
    assert pearson(left, np.asarray(right)) == pytest.approx(expected)


def test_pearson_rejects_constant_input() -> None:
    with pytest.raises(ValueError, match="zero variance"):
        pearson(np.ones(4), np.arange(4.0))


def test_pearson_shared_variance_floor_rejects_near_constant_input() -> None:
    near_constant = 0.5 + np.linspace(0.0, PEARSON_STD_FLOOR, 120)
    assert pearson_input_has_variance(near_constant) is False
    with pytest.raises(ValueError, match="zero variance"):
        pearson(near_constant, np.arange(120.0))


def test_align_pair_uses_sorted_intersection_without_mutation() -> None:
    left = _trace("a", 1, [0.1, 0.2, 0.3, 0.4])
    right = _trace(
        "b",
        1,
        [0.9, 0.8, 0.7, 0.6],
        ids=("d", "c", "b", "extra"),
    )
    left_before = left.dice.copy()
    right_before = right.dice.copy()

    aligned_left, aligned_right = align_pair(left, right, allow_partial=True)

    assert aligned_left.tolist() == [0.2, 0.3, 0.4]
    assert aligned_right.tolist() == [0.7, 0.8, 0.9]
    np.testing.assert_array_equal(left.dice, left_before)
    np.testing.assert_array_equal(right.dice, right_before)


def test_bundle_floor_averages_all_seed_and_case_values() -> None:
    traces = [
        _trace("keep", 1, [0.2, 0.4, 0.2, 0.4]),
        _trace("keep", 2, [0.3, 0.3, 0.3, 0.3]),
        _trace("drop", 1, [0.29, 0.29, 0.29, 0.29]),
        _trace("drop", 2, [0.29, 0.29, 0.29, 0.29]),
    ]

    kept = apply_bundle_floor(traces, floor=0.30)

    assert {trace.bundle for trace in kept} == {"keep"}
    assert len(traces) == 4


def test_within_cross_symmetric_pairing_and_dinov2_cross_checkpoint() -> None:
    traces = []
    patterns = {
        "dinov2_vitb14": [0.0, 1.0, 2.0, 3.0],
        "dinov2_vits14": [0.0, 1.0, 2.0, 3.0],
        "other": [3.0, 2.0, 1.0, 0.0],
    }
    for bundle, values in patterns.items():
        traces.extend([_trace(bundle, seed, values) for seed in (1, 2)])

    result = within_cross(
        traces,
        same_bundle_pairs={frozenset({"dinov2_vitb14", "dinov2_vits14"})},
    )

    assert result["n_within_pairs"] == 7  # 3 C(2,2) + DINO 2x2
    assert result["n_cross_pairs"] == 8  # two remaining bundle pairs, each 2x2
    assert result["within_rbar"] == pytest.approx(1.0)
    assert result["cross_rbar"] == pytest.approx(-1.0)
    assert result["gap"] == pytest.approx(2.0)
