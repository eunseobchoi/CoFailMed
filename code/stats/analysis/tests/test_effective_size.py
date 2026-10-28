from __future__ import annotations

import numpy as np
import pytest

from analysis.effective_size import (
    FIXED_THRESHOLDS,
    UNDEFINED_REASON,
    _pool_spec_diverse4,
    _pool_spec_fixed,
    bootstrap_pool_rows,
    pool_summary,
    threshold_union,
)


def test_threshold_union_contains_exact_sources_and_deduplicates() -> None:
    matrix = np.asarray(
        [
            [0.10, 0.20, 0.30, 0.40, 0.50],
            [0.60, 0.70, 0.80, 0.90, 1.00],
        ]
    )
    rows = threshold_union(matrix)
    sources = {source for row in rows for source in row["sources"]}
    assert sources == {
        "P25",
        "P33",
        "P50",
        *(f"fixed_{value:.2f}" for value in FIXED_THRESHOLDS),
    }
    assert [row["value"] for row in rows] == sorted({row["value"] for row in rows})


def test_support_requires_five_failures_and_five_passes_per_member() -> None:
    supported = np.asarray(
        [
            [0.0] * 5 + [1.0] * 5,
            [1.0, 0.0] * 5,
        ]
    )
    result = pool_summary(supported, 0.5, _pool_spec_fixed(2, "fixed"))
    assert result["status"] == "support-valid"
    assert result["rho_fail"] == pytest.approx(-0.2)
    assert result["M_eff"] == pytest.approx(2.5)

    unsupported = supported.copy()
    unsupported[0, 4] = 1.0
    result = pool_summary(unsupported, 0.5, _pool_spec_fixed(2, "fixed"))
    assert result["status"] == UNDEFINED_REASON
    assert result["rho_fail"] is None
    assert result["M_eff"] is None


def test_diverse4_enumerates_distinct_bundles_one_trace_each() -> None:
    bundles = ("a", "a", "b", "b", "c", "c", "d", "d", "e", "e")
    spec = _pool_spec_diverse4(bundles)
    assert spec.pool_members.shape == (5 * 2**4, 4)
    for pool in spec.pool_members:
        assert len({bundles[index] for index in pool}) == 4


def test_enumerated_summary_is_mean_of_per_pool_meff_not_transform_of_mean_rho() -> None:
    matrix = np.asarray(
        [
            [1, 1, 1, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 0, 0, 1, 1, 1],
            [1, 0, 1, 1, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
        ],
        dtype=float,
    )
    bundles = ("a", "a", "b", "c", "d")
    spec = _pool_spec_diverse4(bundles)
    result = pool_summary(matrix, 0.5, spec)
    assert result["status"] == "support-valid"
    corr = np.corrcoef(matrix)
    per_pool_rho = []
    per_pool_meff = []
    for pool in spec.pool_members:
        rho = np.mean([corr[left, right] for left, right in __import__("itertools").combinations(pool, 2)])
        per_pool_rho.append(rho)
        per_pool_meff.append(4.0 / (1.0 + 3.0 * rho))
    assert result["rho_fail"] == pytest.approx(np.mean(per_pool_rho))
    assert result["M_eff"] == pytest.approx(np.mean(per_pool_meff))
    assert result["M_eff"] != pytest.approx(
        4.0 / (1.0 + 3.0 * result["rho_fail"])
    )


def test_bootstrap_is_deterministic_and_reports_requested_draws() -> None:
    matrix = np.asarray(
        [
            [0] * 5 + [1] * 5,
            [0, 1] * 5,
        ],
        dtype=float,
    )
    spec = _pool_spec_fixed(2, "fixed")
    left = bootstrap_pool_rows(
        matrix, [0.5], spec, b=25, seed=7, cluster_ids=None
    )
    right = bootstrap_pool_rows(
        matrix, [0.5], spec, b=25, seed=7, cluster_ids=None
    )
    assert left == right
    assert left[0.5]["B_requested"] == 25
    assert 0 <= left[0.5]["B_finite"] <= 25




def _case_ids(task: str) -> list[str]:
    if task == "acdc_lv":
        return [f"patient{p:03d}_frame01_slice{s:03d}" for p in range(1, 5) for s in range(3)]
    if task == "brats_wt":
        return [f"BraTS-GLI-{p:05d}-100_slice{s:03d}" for p in range(1, 5) for s in range(3)]
    return [f"{task}_case{index:02d}" for index in range(12)]


@pytest.mark.parametrize("module_name", ["analysis.effective_size", "controls.analysis.effective_size"])
@pytest.mark.parametrize("duplicate_member", [None, 0, 1, 2])
def test_trace_loader_checks_every_members_ids(tmp_path, module_name, duplicate_member) -> None:
    import importlib
    import json

    module = importlib.import_module(module_name)
    ids = [f"case{i}" for i in range(20)]
    values = np.linspace(.1, .9, 20).tolist()
    paths = []
    for member in range(3):
        path = tmp_path / str(member) / "seed_42.json"
        path.parent.mkdir()
        member_ids, member_values = ids.copy(), values.copy()
        if member == duplicate_member:
            member_ids.append(ids[0])
            member_values.append(.9)
        if member == 1:
            member_ids.reverse()
            member_values.reverse()
        path.write_text(json.dumps({"fm": str(member), "test_ids": member_ids,
                                    "per_case_dice": member_values}))
        paths.append(path)
    if duplicate_member is not None:
        with pytest.raises(ValueError, match="duplicate test IDs") as error:
            module._load_trace_paths(paths, config="toy", task="toy", evidence_scope="test")
        assert str(paths[duplicate_member]) in str(error.value)
    else:
        loaded = module._load_trace_paths(paths, config="toy", task="toy", evidence_scope="test")
        assert loaded.case_ids == tuple(ids)
        np.testing.assert_array_equal(loaded.matrix, [values] * 3)
