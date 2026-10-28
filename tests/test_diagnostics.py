"""Point estimates of the appendix diagnostics (diagnostics.py) and small behaviour checks.

Paper constants are transcribed from the final manuscript (Tables C.1, D.1(c), D.2, D.4, D.5, F.1, F.2,
G.1 and G.6, and Appendix C.1 and D.2). Interval checks live in REPRODUCING.md's reference outputs.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import correlation
import diagnostics as dg
from effective_size import enumerate_pools
from scores import floor_panel, load_panel


def by(rows, **match):
    found = [r for r in rows if all(r.get(k) == v for k, v in match.items())]
    if len(found) != 1:
        raise AssertionError(f'{match}: {len(found)} rows')
    return found[0]


class Shown(unittest.TestCase):
    def assertShown(self, value, shown):
        self.assertEqual(f'{value:.{len(shown.split(".")[1])}f}', shown)

    def assertRow(self, row, expected):
        for key, shown in expected.items():
            with self.subTest(key=key):
                self.assertShown(row[key], shown)


class FailureEventTests(Shown):
    def test_table_d2a_points_and_disc_diagnostics(self):
        expected = {'kvasir': ('0.329', '0.072', '-0.176', '0.055'), 'acdc_lv': ('0.449', '0.128', '-0.233', '0.084'),
                    'brats_wt': ('0.469', '0.169', '-0.252', '0.088'), 'riga_cup': ('0.558', '0.438', '-0.282', '0.117'),
                    'riga_disc': ('0.582', '0.547', '-0.362', '0.128')}
        rows = dg.failure_metrics_rows()
        for task, values in expected.items():
            self.assertRow(by(rows, task=task), dict(zip((f'{m}_gap' for m in dg.METRICS), values)))
        disc = by(rows, task='riga_disc')
        self.assertEqual(disc['never_failing'], 'biomedclip/seed_43;biomedclip/seed_45')
        self.assertShown(disc['kappa_gap_without_never_failing'], '0.604')
        self.assertEqual(disc['unsupported_heads'], 'biomedclip/seed_42:1;biomedclip/seed_43:0;biomedclip/seed_44:1;'
                                                    'biomedclip/seed_45:0;dinov2_vitb14/seed_45:4')

    def test_official_kappa_scores_always_agreeing_heads_as_one(self):
        success = np.array([[True] * 6, [True] * 6, [True, False] * 3])
        self.assertEqual(dg._metric_matrices(success)['kappa'][0, 1], 1)
        self.assertTrue(np.isnan(dg._metric_matrices(success, official=False)['kappa'][0, 1]))
        self.assertEqual(dg._metric_matrices(success, official=False)['kappa'][0, 2], 0)


class CalibrationTests(Shown):
    def test_table_d1c_points(self):
        expected = {'kvasir': (42, 78, 10, '0.210', '0.153', '0.836'), 'acdc_lv': (18, 32, 9, '0.269', '0.221', '0.917'),
                    'brats_wt': (51, 95, 11, '0.366', '0.233', '0.901'), 'riga_cup': (33, 62, 10, '0.546', '0.544', '0.713'),
                    'riga_disc': (33, 62, 11, '0.630', '0.572', '0.687')}
        rows = dg.calibration_rows()
        for task, (cal, ev, k, gap, beta, r2) in expected.items():
            row = by(rows, task=task)
            self.assertEqual((row['n_calibration_units'], row['n_evaluation_units'], row['selected_K']), (cal, ev, k))
            self.assertRow(row, {'calibration_gap': gap, 'beta_same': beta, 'r_squared': r2})


class ItemDifficultyTests(Shown):
    def test_table_d5_points(self):
        expected = {'kvasir': ('0.266', '0.683', '0.333', '0.864', '0.129'),
                    'acdc_lv': ('0.328', '0.698', '0.299', '0.895', '0.140'),
                    'brats_wt': ('0.351', '0.686', '0.327', '0.897', '0.134'),
                    'riga_cup': ('0.570', '0.725', '0.518', '0.921', '0.111'),
                    'riga_disc': ('0.655', '0.725', '0.529', '0.940', '0.115')}
        rows = dg.item_difficulty_rows()
        keys = ('raw_gap', 'residual_gap', 'residual_panel_min', 'residual_panel_max', 'residual_panel_sd')
        for task, values in expected.items():
            row = by(rows, task=task)
            self.assertRow(row, dict(zip(keys, values)))
            self.assertEqual((row['n_panels'], row['n_eligible'], row['n_positive']), (70, 70, 70))
        self.assertShown(by(rows, task='acdc_lv')['patient_residual_minus_raw'], '0.235')
        self.assertShown(by(rows, task='brats_wt')['patient_residual_minus_raw'], '0.484')

    def test_closed_form_residual_matches_explicit_regression(self):
        panel = floor_panel(load_panel('primary/riga_cup'))
        setup = dg._panel_setup(list(panel['bundle']))
        families, members, panels = setup
        reference, evaluation, a, b, same = panels[0]
        d = np.mean([panel['dice'][members[f]].mean(axis=0) for f in reference], axis=0)
        design = np.column_stack([np.ones_like(d), d])
        values = panel['dice'][evaluation].T
        residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
        r = np.corrcoef(residual.T)[a, b]
        self.assertAlmostEqual(dg._panel_gaps(panel['dice'], setup)[0, 1], r[same].mean() - r[~same].mean(), places=12)


class TauSweepTests(Shown):
    def test_table_d4_riga_cup_rows(self):
        panel = floor_panel(load_panel('primary/riga_cup'))
        same = np.array([np.flatnonzero(panel['bundle'] == b) for b in np.unique(panel['bundle'])])
        diverse = enumerate_pools(panel)
        expected = {0.70: ('0.087', '0.223'), 0.80: ('0.054', '0.056'), 0.90: ('0.005', '0.001')}
        for tau, (mono, mixed) in expected.items():
            with self.subTest(tau=tau):
                self.assertShown(dg._recovery(same, panel['dice'], tau), mono)
                self.assertShown(dg._recovery(diverse, panel['dice'], tau), mixed)


class ArchitectureTests(Shown):
    def test_table_f1_points(self):
        expected = {'kvasir': ('0.981', '0.774', '0.207'), 'acdc_lv': ('0.993', '0.779', '0.215'),
                    'brats_wt': ('0.988', '0.709', '0.279'), 'riga_cup': ('0.960', '0.358', '0.601'),
                    'riga_disc': ('0.971', '0.242', '0.729')}
        for row in dg.vit_diagnostic_rows():
            self.assertRow(row, dict(zip(('seed_correlation', 'cross_pretraining', 'vit_gap'), expected[row['task']])))
            self.assertEqual((row['n_seed_pairs'], row['n_cross_pretraining_pairs']), (30, 160))

    def test_table_f2_points(self):
        expected = {'kvasir': ('0.061', '0.302', '0.300', '0.735', '0.674', '0.303', 48),
                    'acdc_lv': ('0.150', '0.350', '0.346', '0.786', '0.636', '0.352', 32),
                    'brats_wt': ('0.120', '0.394', '0.391', '0.705', '0.585', '0.396', 32),
                    'riga_cup': ('0.045', '0.643', '0.639', '0.356', '0.311', '0.642', 48),
                    'riga_disc': ('0.165', '0.736', '0.731', '0.390', '0.225', '0.740', 48)}
        keys = ('primary_ckpt_gap', 'gap_same_bundle_only', 'gap_dinov2_as_cross', 'primary_distinct_ckpt',
                'cross_excluding_family', 'same_bundle_gap_vs_cross_excluding_family')
        for row in dg.checkpoints_rows():
            values = expected[row['task']]
            self.assertRow(row, dict(zip(keys, values[:-1])))
            self.assertEqual(row['n_primary_distinct_ckpt_pairs'], values[-1])


class ControlTests(Shown):
    def test_table_g1_nine_bundle_points(self):
        expected = {'kvasir': ('0.173', '0.277', 9), 'acdc_lv': ('0.086', '0.325', 8),
                    'riga_cup': ('0.171', '0.521', 9), 'riga_disc': ('0.264', '0.635', 9)}
        for row in dg.unet_nine_rows():
            unet, frozen, k = expected[row['task']]
            self.assertRow(row, {'unet_gap': unet, 'frozen_gap': frozen})
            self.assertEqual(row['frozen_K'], k)

    def test_table_g6_points(self):
        expected = {('unet_skip', 'riga_cup'): ('0.148', '0.097', '-0.070', '0.025'),
                    ('unet_skip', 'riga_disc'): ('0.226', '0.165', '-0.103', '0.054'),
                    ('nnunet', 'riga_cup'): ('0.099', '0.028', '-0.044', '0.022')}
        for row in dg.pipeline_metrics_rows():
            self.assertTrue(row['all_members_supported'])
            self.assertRow(row, dict(zip((f'{m}_gap' for m in dg.METRICS), expected[row['pipeline'], row['task']])))

    def test_table_c1_points(self):
        rows = dg.bundle_dice_rows()
        self.assertShown(by(rows, task='riga_cup', recipe='frozen', bundle='resnet18')['mean_dice'], '0.304')
        self.assertShown(by(rows, task='riga_cup', recipe='ft_lr1e-5', bundle='efficientnet_b0')['mean_dice'], '0.473')
        below = {(r['task'], r['bundle']) for r in rows if r['recipe'] == 'frozen' and not r['passes_floor']}
        self.assertEqual(below, {('acdc_lv', 'resnet18'), ('brats_wt', 'resnet18')})


class HierarchyTests(Shown):
    def test_patient_aggregated_gaps(self):
        rows = dg.hierarchy_rows()
        self.assertRow(by(rows, task='acdc_lv'), {'patient_gap': '0.493', 'patient_ckpt_gap': '0.315'})
        self.assertRow(by(rows, task='brats_wt'), {'patient_gap': '0.204', 'patient_ckpt_gap': '0.103'})

    def test_self_pair_exclusion_lowers_riga_cup_endpoint(self):
        panel = floor_panel(load_panel('primary/riga_cup'))
        kept = correlation.hierarchical_sensitivity(panel, bootstrap=2000)
        dropped = correlation.hierarchical_sensitivity(panel, bootstrap=2000, drop_self=True)
        self.assertShown(kept['hierarchical_gap_low'], '0.444')
        self.assertShown(dropped['hierarchical_gap_low'], '0.409')


if __name__ == '__main__':
    unittest.main(verbosity=2)
