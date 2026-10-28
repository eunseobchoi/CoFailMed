"""Numerical eligibility in both effective-size scorers and the selected-pool scorer."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from analysis import effective_size as canonical
from analysis import pool_meff

path = Path(canonical.__file__).parents[1] / 'amos/analysis/effective_size.py'
spec = importlib.util.spec_from_file_location('amos_effective_size_eligibility_test', path)
amos = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = amos
spec.loader.exec_module(amos)


@pytest.fixture(params=[canonical, amos], ids=['canonical', 'amos'])
def scorer(request):
    return request.param


def singular():
    return np.array([[0, 1] * 6, [1, 0] * 6] * 2, dtype=float)


def test_singular_support_is_not_eligibility(scorer):
    result = scorer.pool_summary(singular(), .5, scorer._pool_spec_fixed(4, 'fixed'))
    assert result['n_pools_supported'] == 1
    assert result['M_eff'] is None
    assert result['status'] != 'support-valid'
    assert result['n_pools_finite'] == 0
    assert result['n_pools_undefined'] == 1
    assert result['n_pools_numerically_ineligible'] == 1


@pytest.mark.parametrize('denom', [-1e-10, 0., 5e-13, 2e-12, .4, 1., 4.])
def test_denominator_boundary_preserves_formula(scorer, monkeypatch, denom):
    rho = (denom - 1) / 3
    corr = np.full((4, 4), rho)
    np.fill_diagonal(corr, 1)
    monkeypatch.setattr(scorer, '_correlation_matrix', lambda _: corr)
    with np.errstate(divide='raise', invalid='raise'):
        result = scorer.pool_summary(singular(), .5, scorer._pool_spec_fixed(4, 'fixed'))
    if denom <= 1e-12:
        assert result['M_eff'] is None
    else:
        assert result['M_eff'] == pytest.approx(4 / (1 + 3 * np.mean([rho] * 6)))
        assert result['status'] == 'support-valid'


def test_mixed_pool_averages_only_eligible_pools(scorer):
    matrix = np.vstack([singular(), singular()[0]])
    spec = scorer._pool_spec_diverse4(('a', 'b', 'c', 'd', 'd'))
    result = scorer.pool_summary(matrix, .5, spec)
    expected = scorer.pool_summary(matrix[[0, 1, 2, 4]], .5, scorer._pool_spec_fixed(4, 'fixed'))
    assert result['M_eff'] == expected['M_eff']
    assert result['rho_fail'] == expected['rho_fail']
    assert result['n_pools_supported'] == 2
    assert result['n_pools_finite'] == result['n_pools_undefined'] == 1


def test_all_invalid_bootstrap_is_null(scorer, monkeypatch):
    # Preserve the balanced sample to isolate denominator rejection in every draw.
    monkeypatch.setattr(scorer, '_resample_indices', lambda *args: np.arange(12))
    result = scorer.bootstrap_pool_rows(singular(), [.5], scorer._pool_spec_fixed(4, 'fixed'), b=5, seed=0, cluster_ids=None)[.5]
    assert result == {'B_requested': 5, 'B_finite': 0, 'rho_fail_ci95': None, 'M_eff_ci95': None}


def test_actual_split_classifier_and_counts():
    keys = tuple('abcd')
    invalid = pool_meff.split_meff_record(singular(), keys, keys, .5, {}, 0)
    assert invalid['M_eff'] is None and not invalid['low_M_eff']
    assert not invalid['support_valid'] and invalid['support_adequate']
    ordinary = pool_meff.split_meff_record(np.tile(singular()[0], (4, 1)), keys, keys, .5, {}, 1)
    assert ordinary['low_M_eff'] and ordinary['M_eff'] == pytest.approx(1)
    summary = pool_meff.summarize_records([invalid, ordinary], np.array([[0, 0], [0, 1], [1, 1]]))
    assert summary['outcome_counts']['detected_low_M_eff'] == 1
    assert summary['outcome_counts']['unsupported'] == 1
    assert summary['n_numerically_ineligible'] == 1
    assert summary['low_M_eff_fraction_support_valid']['denominator'] == 1
    assert pool_meff.summarize_records([invalid], np.array([[0]]))['support_valid_M_eff_distribution'] is None


def test_actual_split_preserves_valid_negative_rho():
    from itertools import combinations

    columns = [np.isin(np.arange(4), pair).astype(float) for pair in combinations(range(4), 2)]
    matrix = np.column_stack(columns * 5 + [np.zeros(4), np.ones(4)])
    keys = tuple('abcd')
    result = pool_meff.split_meff_record(matrix, keys, keys, .5, {}, 0)
    assert result['support_valid'] and result['support_adequate']
    assert result['rho_fail'] == pytest.approx(-.25)
    assert result['M_eff'] == pytest.approx(16)
    assert result['outcome'] == 'supported_not_low_M_eff'


def test_threshold_record_preserves_numerical_reason(scorer):
    point = scorer.pool_summary(singular(), .5, scorer._pool_spec_fixed(4, 'fixed'))
    row = scorer._threshold_record({'value': .5, 'sources': ['test']}, point, None)
    assert 'denominator' in row['status']
    assert row['M_eff']['value'] is None
    assert row['support']['n_pools_supported'] == 1


def test_all_invalid_thresholds_do_not_claim_missing_support(scorer):
    matrix = singular()
    collection = scorer.TraceCollection(
        config='test', task='test', member_ids=tuple('abcd'), bundle_ids=tuple('abcd'),
        matrix=matrix, case_ids=tuple(map(str, range(12))), cluster_ids=None,
        provenance=(), evidence_scope='test',
    )
    result = scorer.analyze_pool(collection, scorer._pool_spec_fixed(4, 'fixed'), b=5, seed=0)
    supported = [row for row in result['rows'] if row['support']['n_pools_supported']]
    assert supported and all('denominator' in row['status'] for row in supported)
    assert all(row['M_eff']['value'] is None for row in result['rows'])
    assert result['selection']['highest_supported_sensitivity'] is None
