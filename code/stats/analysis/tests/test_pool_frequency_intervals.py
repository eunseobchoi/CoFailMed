"""Fixed-cohort split-frequency intervals must remain nondegenerate at boundaries."""
import pytest

from analysis.pool_low_meff import exact_binomial_interval


def test_all_successes_and_no_successes_have_uncertainty():
    lo, hi = exact_binomial_interval(200, 200)
    assert lo == pytest.approx(0.025 ** (1 / 200))
    assert hi == 1.0 and lo < hi
    lo, hi = exact_binomial_interval(0, 200)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 200))


def test_conditioning_on_support_uses_its_own_denominator():
    assert exact_binomial_interval(3, 140) != exact_binomial_interval(3, 200)
    assert exact_binomial_interval(0, 0) is None


@pytest.mark.parametrize('k,n', [(201, 200), (-1, 200), (1, 0)])
def test_impossible_counts_fail(k, n):
    with pytest.raises(ValueError):
        exact_binomial_interval(k, n)
