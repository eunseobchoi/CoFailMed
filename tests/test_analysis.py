"""Behaviour checks plus displayed paper values; run: python -m unittest discover -s tests -v.

Requires only the release's NumPy and SciPy dependencies and stdlib unittest. Paper constants
below are transcribed from the final manuscript tables, not from these routines.
"""
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import control_comparisons as controls
import correlation
import effective_size as es
import sensitivity
from scores import CASE_FIELDS, PRIMARY_TASKS, floor_panel, load_panel, select_rows


def panel(dice, bundles=None):
    dice = np.asarray(dice, dtype=float)
    m, n = dice.shape
    bundles = np.asarray(bundles if bundles is not None else [f'b{i // 2}' for i in range(m)])
    ids = np.asarray([f'case{i}' for i in range(n)])
    return dict(dice=dice, bundle=bundles, seed=np.arange(m), case_ids=ids,
                cluster_ids=ids.copy(), upstream_family=bundles.copy(),
                selection_family=bundles.copy())


def permute_members(p, order):
    return {k: v.copy() if k in CASE_FIELDS else v[order].copy() for k, v in p.items()}


def hierarchy_fixture():
    p = panel([[.1, .3, .7, .9, .2, .6, .4], [.2, .6, .4, .8, .5, .1, .7],
               [.9, .7, .3, .1, .8, .4, .6], [.8, .4, .6, .2, .5, .9, .3],
               [.1, .2, .3, .4, .5, .6, .7], [.7, .6, .5, .4, .3, .2, .1]],
              ['dinov2_vitb14', 'dinov2_vits14', 'a', 'a', 'z', 'z'])
    p['seed'] = np.array([42, 42, 42, 43, 42, 43])
    return p


class HierarchyTests(unittest.TestCase):
    def test_original_study_collapse_duplicates_and_patient_multiplicity(self):
        # Original fmpool.estimators.hierarchical_bootstrap, seed=3, B=64,
        # with compute_all's DINOv2-S +10000 seed relabeling. These fixtures
        # include repeated heads and repeated group occurrences.
        for clustered, expected in [(False, (-.46995532486704156, 1.805938947056569)),
                                    (True, (-.7071769895471787, 1.580964926526734))]:
            with self.subTest(clustered=clustered):
                p = hierarchy_fixture()
                if clustered:
                    p['cluster_ids'] = np.array(['q', 'q', 'q', 'r', 'r', 's', 's'])
                result = correlation.hierarchical_sensitivity(p, bootstrap=64, seed=3)
                self.assertEqual(result['hierarchical_n_valid'], 57)
                self.assertEqual(result['hierarchical_unit'], 'patient' if clustered else 'case')
                np.testing.assert_allclose([result['hierarchical_gap_low'], result['hierarchical_gap_high']],
                                           expected, atol=1e-13, rtol=0)
                shuffled = permute_members(p, np.arange(6)[::-1])
                self.assertEqual(result, correlation.hierarchical_sensitivity(shuffled, bootstrap=64, seed=3))

    def test_no_valid_draws_stay_nan(self):
        p = panel(np.full((4, 6), .5))
        result = correlation.hierarchical_sensitivity(p, bootstrap=16)
        self.assertEqual(result['hierarchical_n_valid'], 0)
        self.assertTrue(np.isnan(result['hierarchical_gap_low']))
        self.assertTrue(np.isnan(result['hierarchical_gap_high']))

    def test_rejects_one_collapsed_group_and_single_patient(self):
        p = hierarchy_fixture()
        with self.assertRaisesRegex(ValueError, 'release groups'):
            correlation.hierarchical_sensitivity(select_rows(p, np.arange(6) < 2))
        p['cluster_ids'][:] = 'one'
        with self.assertRaisesRegex(ValueError, 'two patients'):
            correlation.hierarchical_sensitivity(p)
        with self.assertRaisesRegex(ValueError, 'unit'):
            correlation.hierarchical_sensitivity(hierarchy_fixture(), unit='invalid')
        with self.assertRaises(ValueError):
            correlation.hierarchical_sensitivity(hierarchy_fixture(), unit='patient')

    def test_malformed_inputs_and_bootstrap_are_rejected(self):
        for value in (0, -1, True, np.bool_(True), 1.5):
            with self.subTest(bootstrap=value), self.assertRaises(ValueError):
                correlation.hierarchical_sensitivity(hierarchy_fixture(), bootstrap=value)
        mutations = [('dice', np.full((6, 7), np.nan)), ('dice', np.full((6, 7), 1.01)),
                     ('dice', np.ones((6, 1))), ('seed', np.ones(6)),
                     ('case_ids', np.array(['duplicate'] * 7)),
                     ('cluster_ids', np.array(['short'])), ('bundle', np.array([''] * 6))]
        for field, value in mutations:
            with self.subTest(field=field, shape=value.shape), self.assertRaises(ValueError):
                p = hierarchy_fixture()
                p[field] = value
                correlation.hierarchical_sensitivity(p, bootstrap=2)
        p = hierarchy_fixture()
        p['seed'][3] = p['seed'][2]
        with self.assertRaisesRegex(ValueError, 'duplicate member'):
            correlation.hierarchical_sensitivity(p, bootstrap=2)


class EffectiveSizeTests(unittest.TestCase):
    def test_independent_identical_and_complementary_pools(self):
        independent = np.array(list(itertools.product([0, 1], repeat=4))).T
        identical = np.tile([0] * 5 + [1] * 5, (4, 1))
        complementary = identical.copy()
        complementary[2:] = 1 - complementary[2:]
        for fail, expected in [(independent, 4), (identical, 1), (complementary, np.nan)]:
            with self.subTest(expected=expected):
                got = es._pool_summary(np.where(fail, .5, .85), .85,
                                       np.arange(4)[None, :], separate=True)
                np.testing.assert_allclose([got['M_eff'], got['M_eff_covariance_exact']],
                                           [expected, expected], atol=1e-12, equal_nan=True)
                self.assertEqual(got['n_supported'], 1)

    def test_fixed_covariance_has_independent_eligibility(self):
        raw_zero = np.equal.outer(np.arange(4), np.repeat(np.arange(4), [5, 5, 10, 20]))
        partition = np.equal.outer(np.arange(3), np.repeat(np.arange(3), [35, 35, 5]))
        standardized_zero = partition[[0, 0, 1, 1, 2]]
        for fail, eligible in [(raw_zero, (1, 0)), (standardized_zero, (0, 1))]:
            with self.subTest(eligible=eligible):
                m = len(fail)
                dice, pools = np.where(fail, .5, .85), np.arange(m)[None, :]
                got = es._pool_summary(dice, .85, pools, separate=True)
                # Independent sample covariance reference, not production's scatter calculation.
                covariance = np.cov(fail.astype(float), ddof=1)
                corr = covariance / np.sqrt(np.outer(covariance.diagonal(), covariance.diagonal()))
                denominator = 1 + (m - 1) * corr[np.triu_indices(m, 1)].mean()
                expected = m / denominator if eligible[0] else np.nan
                exact = m * covariance.trace() / covariance.sum() if eligible[1] else np.nan
                np.testing.assert_allclose([got['M_eff'], got['M_eff_covariance_exact']],
                                           [expected, exact], atol=1e-11, equal_nan=True)
                self.assertEqual((got['n_eligible'], got['n_covariance_eligible']), eligible)
                self.assertEqual(got['n_joint_eligible'], 0)
                self.assertEqual(got['n_rho_eligible'], 1)
                joint = es._pool_summary(dice, .85, pools, covariance=True)
                self.assertEqual(joint['n_covariance_eligible'], 0)
                self.assertTrue(np.isnan(joint['M_eff_covariance_exact']))

    def test_support_boundary_ties_empty_pools_and_bad_scores(self):
        dice = np.tile([.5] * 5 + [.85] * 5, (4, 1))
        self.assertAlmostEqual(es.effective_size(dice, .85), 1)
        dice[0, 0] = .85
        self.assertTrue(np.isnan(es.effective_size(dice, .85)))
        got = es._pool_summary(dice, .85, np.empty((0, 4), dtype=int), separate=True)
        self.assertEqual(got['n_pools'], 0)
        self.assertTrue(np.isnan(got['M_eff_covariance_exact']))
        for bad in (np.ones(5), np.full((4, 10), np.nan), np.full((4, 10), -1)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                es.effective_size(bad, .85)
        for threshold in (np.nan, -1, 2):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                es.effective_size(dice, threshold)


class ControlTests(unittest.TestCase):
    def test_paired_zero_change_and_member_permutation(self):
        p = hierarchy_fixture()
        for other in (p, permute_members(p, np.arange(6)[::-1])):
            got = controls.paired_gap(other, p, bootstrap=32)
            self.assertEqual(got['gap_difference_n_valid'], 32)
            for key in ('gap_difference', 'gap_difference_ci95_low', 'gap_difference_ci95_high'):
                self.assertAlmostEqual(got[key], 0, places=13)

    def test_paired_rejects_unaligned_identities(self):
        p = hierarchy_fixture()
        for key in ('case_ids', 'cluster_ids', 'case_split_ids', 'scored_export_ids', 'seed'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                other = {k: v.copy() for k, v in p.items()}
                other[key] = (other[key][::-1] if key in other else np.array(['x'] * 7))
                controls.paired_gap(other, p)
        other = {k: v.copy() for k, v in p.items()}
        for key in CASE_FIELDS & other.keys():
            other[key] = other[key][::-1]
        other['dice'] = other['dice'][:, ::-1]
        with self.assertRaises(ValueError):
            controls.paired_gap(other, p)
        for value in (-1, True, np.bool_(True), 1.5):
            with self.subTest(bootstrap=value), self.assertRaises(ValueError):
                controls.paired_gap(p, p, bootstrap=value)

    def test_recipe_specific_floors(self):
        p = hierarchy_fixture()
        other = {k: v.copy() for k, v in p.items()}
        other['dice'][other['bundle'] == 'a'] *= .1
        got = controls.paired_gap(other, p)
        self.assertEqual((got['K_ft'], got['K_frozen']), (3, 4))
        self.assertNotIn('a', got['kept_ft'].split(';'))
        self.assertIn('a', got['kept_frozen'].split(';'))

    def test_nnunet_uses_seed13_and_requires_exact_folds(self):
        x = np.linspace(.1, .9, 40)
        p = panel(np.tile(x, (10, 1)), ['nnunet'] * 10)
        p['seed'] = np.repeat([13, 37], 5)
        p['fold'] = np.tile(np.arange(5), 2)
        p['dice'][5:] = 1 - p['dice'][5:]
        got = controls.nnunet_size(p, seed=0, clustered=False, bootstrap=16)
        self.assertAlmostEqual(got['M_eff'], 1)
        self.assertAlmostEqual(got['pooled_P33'], 11 / 30)  # P33 of five repeated 40-case traces.
        self.assertEqual(got['M_eff_n_valid'], 16)
        for key in ('M_eff_ci95_low', 'M_eff_ci95_high'):
            self.assertAlmostEqual(got[key], 1)
        for bad in (select_rows(p, np.arange(10) != 0), {k: v for k, v in p.items() if k != 'fold'}):
            with self.assertRaises(ValueError):
                controls.nnunet_size(bad, seed=0, clustered=False)


class SensitivityTests(unittest.TestCase):
    def test_icc_absolute_agreement_and_degeneracy(self):
        x = np.array([0, .2, .4])
        self.assertAlmostEqual(sensitivity._icc_a1(x, x), 1)
        self.assertAlmostEqual(sensitivity._icc_a1(x, x + .2), 2 / 3)
        self.assertTrue(np.isnan(sensitivity._icc_a1(np.ones(3), np.ones(3))))
        self.assertTrue(np.isnan(sensitivity._icc_a1(x[:1], x[:1])))

    def test_ols_against_independent_full_rank_normal_equations(self):
        p = panel(np.random.default_rng(19).uniform(.2, .9, (8, 20)))
        p['dice'][-1] = .5
        x, y = [], []
        for a, b in itertools.combinations(range(7), 2):
            u, v = p['dice'][a], p['dice'][b]
            du, dv = u - u.mean(), v - v.mean()
            y.append(float(du @ dv / np.sqrt((du @ du) * (dv @ dv))))
            x.append([1, p['bundle'][a] == p['bundle'][b], (u.mean() + v.mean()) / 2,
                      abs(u.mean() - v.mean()), (np.mean(du ** 2) + np.mean(dv ** 2)) / 2,
                      abs(np.mean(du ** 2) - np.mean(dv ** 2))])
        x, y = np.array(x), np.array(y)
        self.assertEqual(np.linalg.matrix_rank(x), 6)
        beta = np.linalg.solve(x.T @ x, x.T @ y)
        expected_r2 = 1 - np.sum((y - x @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
        got = sensitivity._quality_stats(p)
        self.assertAlmostEqual(got['beta_same'], beta[1], places=10)
        self.assertAlmostEqual(got['r_squared'], expected_r2, places=10)
        self.assertEqual(got['n_regression_pairs'], 21)
        with self.assertRaises(ValueError):
            sensitivity._quality_stats(panel(np.full((4, 10), .5)))

    def test_failure_ties_constant_members_and_joint_rates(self):
        p = panel([[0, .5, 1, .5], [0, .5, 1, .5], [.5, .5, .5, .5], [1, .5, 0, .5]])
        got = sensitivity._failure_stats(p, 50)
        expected = dict(threshold=.5, n_values_equal_threshold=10, n_constant_failure_members=1,
                        n_same_pairs=2, n_cross_pairs=4, n_same_defined_pairs=1,
                        n_cross_defined_pairs=2, same_fail_phi=1, cross_fail_phi=-1 / 3,
                        gap_fail_phi=4 / 3, same_joint_fail_rate=1 / 8, cross_joint_fail_rate=0)
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertAlmostEqual(got[key], value)
        self.assertTrue(np.isnan(got['joint_fail_rate_ratio']))


class BoundaryRegressionTests(unittest.TestCase):
    def test_fixed_and_p33_reject_truncated_member_metadata(self):
        x = np.array([.4] * 5 + [.9] * 5)
        p = panel([x, x, x, x, np.roll(x, 2)], list('abcde'))
        p['bundle'] = p['bundle'][:4]
        for summarize in (es.fixed_panel_summary, es.panel_summary):
            with self.subTest(function=summarize.__name__), self.assertRaises(ValueError):
                summarize(p)

    def test_paired_rejects_fold_bearing_panels(self):
        x = np.array([.4] * 5 + [.9] * 5)
        p = panel([np.roll(x, k) for k in range(4)], ['a', 'a', 'b', 'b'])
        p['seed'][:] = 42
        p['fold'] = np.array([0, 1, 0, 1])
        other = {k: v.copy() for k, v in p.items()}
        other['fold'] += 2
        for candidate in (p, other):
            with self.assertRaisesRegex(ValueError, 'fold'):
                controls.paired_gap(candidate, p)

    def test_hierarchy_rejects_reserved_group_alias(self):
        p = hierarchy_fixture()
        for field in ('bundle', 'upstream_family', 'selection_family'):
            p[field][p[field] == 'a'] = 'dinov2'
        with self.assertRaisesRegex(ValueError, 'dinov2'):
            correlation.hierarchical_sensitivity(p, bootstrap=8)

    def test_nnunet_rejects_mixed_bundle_folds(self):
        p = panel(np.tile(np.linspace(.1, 1, 20), (5, 1)), list('abcde'))
        p['seed'][:] = 13
        p['fold'] = np.arange(5)
        with self.assertRaisesRegex(ValueError, 'bundle'):
            controls.nnunet_size(p, seed=0, clustered=False)


class PaperPointTests(unittest.TestCase):
    """Final TeX: Tables D.1/D.3/E.1(a)/G.3/G.5, rounded displayed values only."""
    @classmethod
    def setUpClass(cls):
        cls.rows = sensitivity.analysis_rows()

    def test_sensitivity_displayed_cells(self):
        # Table D.1(b) and D.1(c), task order below.
        icc = (.334, .397, .435, .596, .716)
        beta = (.153, .221, .233, .544, .572)
        r2 = (.836, .917, .901, .713, .687)
        # Table D.3: P33 threshold, phi gap, joint ratio.
        failures = ((.569, .290, 1.259), (.547, .405, 1.427), (.616, .425, 1.498),
                    (.521, .537, 1.800), (.858, .576, 2.071))
        for index, task in enumerate(PRIMARY_TASKS):
            rows = {r['analysis']: r for r in self.rows if r['task'] == task and r.get('percentile', 33) == 33}
            for actual, expected in [(rows['icc']['gap'], icc[index]), (rows['quality']['beta_same'], beta[index]),
                                     (rows['quality']['r_squared'], r2[index])]:
                with self.subTest(task=task, expected=expected):
                    self.assertEqual(f'{actual:.3f}', f'{expected:.3f}')
            for key, expected in zip(('threshold', 'gap_fail_phi', 'joint_fail_rate_ratio'), failures[index]):
                self.assertEqual(f'{rows["threshold"][key]:.3f}', f'{expected:.3f}')

    def test_floor_and_leave_out_displayed_cells(self):
        # Table D.1(a) and D.1(b), all five primary tasks.
        floor_gaps = ((.265, .265, .265, .265, .265, .225),
                      (.351, .351, .351, .317, .317, .227),
                      (.382, .382, .382, .340, .340, .340),
                      (.557, .557, .557, .557, .541, .531),
                      (.640, .640, .640, .640, .640, .640))
        leave_out = ((.225, .308, .464), (.278, .361, .451), (.314, .399, .445),
                     (.535, .653, .391), (.602, .750, .370))
        for task, gaps, omissions in zip(PRIMARY_TASKS, floor_gaps, leave_out):
            for floor, expected in zip(sensitivity.FLOORS, gaps):
                row = next(r for r in self.rows if r['task'] == task and r['analysis'] == 'floor'
                           and r['mean_dice_floor'] == floor)
                with self.subTest(task=task, floor=floor):
                    self.assertEqual(f'{row["gap"]:.3f}', f'{expected:.3f}')
            row = next(r for r in self.rows if r['task'] == task and r['analysis'] == 'leave_out_summary')
            for key, expected in zip(('leave_one_bundle_out_gap_min', 'leave_one_bundle_out_gap_max',
                                      'leave_one_task_out_mean_gap'), omissions):
                self.assertEqual(f'{row[key]:.3f}', f'{expected:.3f}')

    def test_fixed_displayed_cells(self):
        # Table E.1(a): Eq.2, Eq.3, supported/total pools.
        expected = ((1.865, 1.864, 53760, 84480), (2.360, 2.313, 53760, 53760),
                    (2.154, 2.178, 53760, 53760), (2.309, 2.317, 13440, 84480),
                    (3.001, 3.055, 48384, 84480))
        for task, (eq2, eq3, supported, total) in zip(PRIMARY_TASKS, expected):
            with self.subTest(task=task):
                got = es.fixed_panel_summary(floor_panel(load_panel('primary/' + task)))
                self.assertEqual(f'{got["M_eff"]:.3f}', f'{eq2:.3f}')
                self.assertEqual(f'{got["M_eff_covariance_exact"]:.3f}', f'{eq3:.3f}')
                self.assertEqual((got['n_supported'], got['n_pools']), (supported, total))

    def test_control_displayed_cells(self):
        # Table G.3 paired changes and Table G.5 nnU-Net values.
        expected = {('kvasir', 'ft_lr1e-4'): '-0.083', ('kvasir', 'ft_lr1e-5'): '-0.027',
                    ('riga_cup', 'ft_lr1e-4'): '-0.344', ('riga_cup', 'ft_lr1e-5'): '-0.274'}
        for row in controls.analysis_rows():
            with self.subTest(task=row['task'], recipe=row['recipe']):
                if row['table'] == 'G.3':
                    self.assertEqual(f'{row["gap_difference"]:.3f}', expected[row['task'], row['recipe']])
                else:
                    size, dice = ('1.244', '.942') if row['task'] == 'acdc_lv' else ('1.264', '.863')
                    self.assertEqual(f'{row["M_eff"]:.3f}', size)
                    self.assertEqual(f'{row["mean_dice"]:.3f}', '0' + dice)

    def test_headline_effective_sizes_and_pool_selection(self):
        # Headline values: P33 effective sizes (Table E.1(b)) and 910/1,400 low-M_eff splits (Table I.1(b)).
        import pool_selection as ps
        expected = dict(zip(PRIMARY_TASKS, ('1.559', '1.645', '1.828', '2.420', '2.918')))
        for task in PRIMARY_TASKS:
            with self.subTest(task=task):
                row = es.panel_summary(floor_panel(load_panel('primary/' + task)))
                self.assertEqual(f"{row['M_eff']:.3f}", expected[task])
        records = [r for t in ps.TASKS for r in ps.selection_records(
            load_panel(('primary/' if t in PRIMARY_TASKS else 'heldout/') + t))]
        summary = ps.summarize('all_tasks', records)
        self.assertEqual((summary['low_M_eff'], summary['n_splits'], summary['unsupported']), (910, 1400, 61))


if __name__ == '__main__':
    unittest.main(verbosity=2)
