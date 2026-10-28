"""Appendix-extension checks, release manifest and Croissant metadata.

Run with the other tests: python -m unittest discover -s tests -v. Paper constants below are
transcribed from the final manuscript (Tables B.2, C.2-C.5, G.2, G.3, G.7, G.8, H.2 and I.1(a)).
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import correlation
import extensions as ext
from scores import PACKS, floor_panel, load_amos, load_panel, load_pixel_overlap

RAW = 'https://raw.githubusercontent.com/eunseobchoi/CoFailMed/main/'


def by(rows, **match):
    found = [r for r in rows if all(r.get(k) == v for k, v in match.items())]
    if len(found) != 1:
        raise AssertionError(f'{match}: {len(found)} rows')
    return found[0]


class DisplayedValueMixin:
    def assertShown(self, value, shown):
        decimals = len(shown.split('.')[1])
        self.assertEqual(f'{value:.{decimals}f}', shown)


class MaskAlignTests(DisplayedValueMixin, unittest.TestCase):
    def test_table_b2_points(self):
        # Task: (K, gap nearest-exact, exact minus nearest).
        expected = {'kvasir': (11, '0.267', '0.0018'), 'acdc_lv': (10, '0.322', '0.0043'),
                    'riga_cup': (11, '0.554', '-0.0029'), 'riga_disc': (11, '0.639', '-0.0008')}
        rows = ext.mask_align_rows()
        for task, (k, gap, diff) in expected.items():
            with self.subTest(task=task):
                row = by(rows, task=task)
                self.assertEqual((row['K_nearest'], row['K_exact']), (k, k))
                self.assertShown(row['gap_exact'], gap)
                self.assertShown(row['gap_difference'], diff)
        self.assertShown(max(abs(r['gap_difference']) for r in rows), '0.0043')

    def test_paired_draws_follow_the_released_correlation_stream(self):
        nearest = floor_panel(load_panel('primary/riga_cup'))
        masks = correlation._pair_masks(nearest, 'release')
        gaps = [correlation._point(nearest['dice'][:, idx], masks)[2] for idx in ext._case_draws(nearest, 40)]
        reference = correlation.correlate(nearest, bootstrap=40)
        np.testing.assert_allclose(np.percentile(gaps, [2.5, 97.5]),
                                   [reference['gap_ci_low'], reference['gap_ci_high']], rtol=0, atol=1e-12)


class PerRaterTests(DisplayedValueMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = ext.per_rater_rows()

    def test_table_c2_gt_specific_points(self):
        cup = ('0.545', '0.509', '0.539', '0.511', '0.526', '0.516', '0.557')
        disc = ('0.647', '0.627', '0.623', '0.651', '0.627', '0.621', '0.640')
        references = [f'rater_{k}' for k in range(1, 7)] + ['consensus']
        for task, gaps, ks in (('riga_cup', cup, (10, 10, 10, 10, 11, 11, 11)),
                               ('riga_disc', disc, (11,) * 7)):
            for reference, gap, k in zip(references, gaps, ks):
                with self.subTest(task=task, reference=reference):
                    row = by(self.rows, task=task, pool='gt_specific', reference=reference)
                    self.assertShown(row['gap'], gap)
                    self.assertEqual((row['K'], row['n_members']), (k, 4 * k))

    def test_fixed_pool_and_consensus_summaries(self):
        fixed = ('0.563', '0.536', '0.550', '0.537', '0.526', '0.516')
        for k, gap in enumerate(fixed, 1):
            self.assertShown(by(self.rows, task='riga_cup', pool='fixed_consensus', reference=f'rater_{k}')['gap'], gap)
        cup = by(self.rows, task='riga_cup', pool='gt_specific', reference='consensus')
        self.assertShown(cup['consensus_minus_single_rater_max'], '0.012')
        dropped = [r['reference'] for r in self.rows if r['task'] == 'riga_cup' and r['pool'] == 'gt_specific'
                   and r['excluded_bundles'] == 'resnet18']
        self.assertEqual(dropped, ['rater_1', 'rater_2', 'rater_3', 'rater_4'])
        primary = correlation.correlate(floor_panel(load_panel('primary/riga_cup')))
        self.assertEqual(cup['gap'], primary['gap'])


class KvasirGroupedTests(DisplayedValueMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = ext.kvasir_grouped_rows()

    def test_table_c5_points(self):
        expected = {('official', 'primary'): (11, '0.940', '0.675', '0.265'),
                    ('content_group', 'primary'): (11, '0.933', '0.610', '0.323'),
                    ('official', 'distinct_checkpoint'): (17, None, None, '0.086'),
                    ('content_group', 'distinct_checkpoint'): (17, None, None, '0.117'),
                    ('official', 'unet_skip'): (11, None, None, '0.158'),
                    ('content_group', 'unet_skip'): (11, None, None, '0.173')}
        for (split, analysis), (k, within, cross, gap) in expected.items():
            with self.subTest(split=split, analysis=analysis):
                row = by(self.rows, table='C.5', split=split, analysis=analysis)
                self.assertEqual(row['K'], k)
                self.assertShown(row['gap'], gap)
                if within:
                    self.assertShown(row['within'], within)
                    self.assertShown(row['cross'], cross)
                self.assertEqual(row['n_test_images_shared_by_splits'], 12)

    def test_tables_g2_g3_content_group_points(self):
        for recipe, (within, cross, gap, change) in {'ft_lr1e-4': ('0.776', '0.601', '0.175', '-0.148'),
                                                     'ft_lr1e-5': ('0.884', '0.657', '0.227', '-0.096')}.items():
            with self.subTest(recipe=recipe):
                row = by(self.rows, analysis='fine_tuning', recipe=recipe)
                for key, shown in (('within', within), ('cross', cross), ('gap', gap),
                                   ('gap_difference', change), ('frozen_gap', '0.323')):
                    self.assertShown(row[key], shown)
                self.assertEqual((row['K'], row['K_frozen']), (11, 11))
        self.assertShown(by(self.rows, analysis='fine_tuning_excluding_collapsed_head')['gap'], '0.177')
        lower = {(r['split'], r['bundle']) for r in self.rows if r.get('analysis') == 'bundle_dice'
                 and r['recipe'] == 'ft_lr1e-4' and r['ft_minus_frozen'] < 0}
        self.assertEqual(lower, {('official', 'dinov2_vitb14'), ('official', 'dinov2_vits14'),
                                 ('content_group', 'dinov2_vitb14'), ('content_group', 'dinov2_vits14'),
                                 ('content_group', 'clip_vitl14')})

    def test_content_group_cases_are_the_committed_test_split(self):
        split = json.loads((ROOT / 'code/data/splits/kvasir_grouped_split.json').read_text())
        self.assertEqual(sorted(split['test']), sorted(load_panel('kvasir_grouped/primary')['case_ids']))


class MedicalEncoderTests(DisplayedValueMixin, unittest.TestCase):
    def test_table_h2_released_rows(self):
        # (task, encoder): mean Dice, rho to primary, rho_cross, delta_med.
        expected = {('kvasir', 'retfound_vitl16'): ('0.515', '0.597', '0.675', '-0.078'),
                    ('kvasir', 'rad_dino'): ('0.571', '0.593', '0.675', '-0.082'),
                    ('kvasir', 'biomedparse'): ('0.715', '0.592', '0.675', '-0.083'),
                    ('riga_cup', 'retfound_vitl16'): ('0.740', '0.357', '0.310', '0.048'),
                    ('riga_cup', 'rad_dino'): ('0.525', '0.339', '0.310', '0.029'),
                    ('riga_cup', 'biomedparse'): ('0.743', '0.323', '0.310', '0.013'),
                    ('riga_disc', 'retfound_vitl16'): ('0.936', '0.040', '0.230', '-0.190'),
                    ('riga_disc', 'rad_dino'): ('0.895', '0.212', '0.230', '-0.018'),
                    ('riga_disc', 'biomedparse'): ('0.900', '0.131', '0.230', '-0.099')}
        rows = ext.medical_encoders_rows()
        self.assertEqual(len(rows), len(expected))
        for (task, encoder), shown in expected.items():
            row = by(rows, task=task, encoder=encoder)
            for key, value in zip(('mean_dice', 'rho_to_primary', 'cross', 'delta_med'), shown):
                with self.subTest(task=task, encoder=encoder, key=key):
                    self.assertShown(row[key], value)
        tops = {r['task']: r['top_primary_bundle'] for r in rows if r['encoder'] == 'rad_dino'}
        self.assertEqual(tops['riga_cup'], 'dinov2_vitb14')
        self.assertNotIn('dinov2_vitb14', (tops['riga_disc'], tops['kvasir']))

    def test_no_phikon_values_are_packed(self):
        with np.load(ROOT / 'data/extensions.npz', allow_pickle=False) as pack:
            for key in pack.files:
                if pack[key].dtype.kind == 'U':
                    self.assertFalse(any('phikon' in str(x).lower() for x in pack[key]), key)


class AmosTests(DisplayedValueMixin, unittest.TestCase):
    def test_tables_g7_g8_points_and_organ_counts(self):
        expected = {'ct': ('0.988', '0.976', '0.012', '1.071', '0.798', '0.038', '0.004', '-0.017', '0.008'),
                    'mri': ('0.975', '0.918', '0.057', '1.268', '0.748', '0.143', '0.009', '-0.083', '-0.033')}
        keys = ('same_r', 'cross_r', 'gap', 'M_eff', 'pooled_P33', 'kappa_gap', 'q_gap',
                'disagreement_gap', 'double_fault_gap')
        rows = ext.amos_rows()
        for modality, shown in expected.items():
            row = by(rows, modality=modality)
            self.assertEqual((row['n_cases'], row['n_organs']), (60, 13))
            for key, value in zip(keys, shown):
                with self.subTest(modality=modality, key=key):
                    self.assertShown(row[key], value)
        self.assertEqual(sum(r['undefined_organ_cells_per_model'] for r in rows), 14)
        self.assertEqual((min(r['min_finite_organs_per_case'] for r in rows),
                          max(r['max_finite_organs_per_case'] for r in rows)), (11, 13))

    def test_failure_gaps_match_pairwise_definitions(self):
        success = np.random.default_rng(5).random((3, 40)) > 0.4
        got = ext._failure_gaps(success)
        def pair(a, b):
            n11, n00 = np.sum(a & b), np.sum(~a & ~b)
            n10, n01 = np.sum(a & ~b), np.sum(~a & b)
            p, q, n = a.mean(), b.mean(), len(a)
            expected = p * q + (1 - p) * (1 - q)
            return {'kappa': ((n11 + n00) / n - expected) / (1 - expected),
                    'q': (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10),
                    'disagreement': (n01 + n10) / n, 'double_fault': n00 / n}
        same, crosses = pair(success[0], success[1]), [pair(success[i], success[2]) for i in (0, 1)]
        for key in got:
            self.assertAlmostEqual(got[key], same[key] - np.mean([c[key] for c in crosses]), places=12)

    def test_amos_loader_checks_case_scores(self):
        panel = load_amos('ct')
        np.testing.assert_allclose(np.nanmean(panel['organ_dice'], axis=2), panel['dice'], rtol=0, atol=1e-12)
        with self.assertRaises(ValueError):
            load_panel('amos/ct')
        with tempfile.TemporaryDirectory() as tmp:
            bad = {f'amos/ct/{k}': v for k, v in panel.items()}
            bad['amos/ct/dice'] = panel['dice'] * 0.5
            np.savez(Path(tmp) / 'extensions.npz', **bad)
            entry = next(p for p in json.loads((ROOT / 'data/metadata.json').read_text())['panels']
                         if p['key'] == 'amos/ct')
            (Path(tmp) / 'metadata.json').write_text(json.dumps({'panels': [entry]}))
            with self.assertRaisesRegex(ValueError, 'finite organ'):
                load_amos('ct', tmp)


class HeldoutTests(DisplayedValueMixin, unittest.TestCase):
    def test_table_i1a_points(self):
        # Task: K/available, rho_same, rho_cross, gap, N (Table I.1(a)).
        expected = {'kvasir': (11, '0.940', '0.675', '0.265', 120), 'acdc_lv': (10, '0.955', '0.638', '0.317', 977),
                    'brats_wt': (10, '0.927', '0.587', '0.340', 3365), 'riga_cup': (11, '0.867', '0.310', '0.557', 95),
                    'riga_disc': (11, '0.870', '0.230', '0.640', 95), 'isic2018': (11, '0.930', '0.679', '0.250', 1000),
                    'msd_hippocampus': (11, '0.908', '0.441', '0.467', 52),
                    'msd_spleen': (11, '0.983', '0.979', '0.003', 8)}
        rows = ext.heldout_rows()
        for task, (k, within, cross, gap, n) in expected.items():
            with self.subTest(task=task):
                row = by(rows, task=task)
                self.assertEqual((row['K'], row['K_available'], row['n_cases']), (k, 11, n))
                for key, shown in (('within', within), ('cross', cross), ('gap', gap)):
                    self.assertShown(row[key], shown)
        heart = by(rows, task='msd_heart')
        self.assertEqual((heart['K'], heart['K_available'], heart['n_cases']), (0, 11, 4))
        self.assertNotIn('gap', heart)

    def test_spleen_degenerate_gap_and_empty_masks(self):
        spleen = load_panel('heldout/msd_spleen')
        row = correlation.correlate(floor_panel(spleen), bootstrap=2000)
        self.assertEqual(row['n_valid_boot'], 1987)
        self.assertAlmostEqual(row['gap_ci_low'], 0, places=12)  # printed 0.000
        self.assertShown(row['gap_ci_high'], '0.007')
        counts = ext._empty_mask_counts(spleen)
        self.assertEqual((counts['n_target_empty'], len(spleen['case_ids'])), (4, 8))
        self.assertEqual((counts['n_heads_without_foreground'], len(spleen['dice'])), (38, 44))
        self.assertEqual((counts['n_nonempty_predictions'], counts['n_predictions']), (6, 352))
        self.assertEqual((counts['n_empty_target_dice_one'], counts['n_empty_target_pairs']), (174, 176))
        self.assertEqual((counts['n_nonempty_target_dice_zero'], counts['n_nonempty_target_pairs']), (176, 176))

    def test_empty_mask_flags_must_agree_with_dice(self):
        spleen = load_panel('heldout/msd_spleen')
        broken = {k: v.copy() for k, v in spleen.items()}
        broken['prediction_empty'][:, 0] = ~broken['prediction_empty'][:, 0]
        with self.assertRaisesRegex(ValueError, 'empty-mask convention'):
            floor_panel(broken)
        with self.assertRaisesRegex(ValueError, 'together'):
            floor_panel({k: v for k, v in spleen.items() if k != 'target_empty'})

    def test_msd_cases_are_the_committed_test_splits(self):
        for task in ('msd_spleen', 'msd_heart'):
            split = json.loads((ROOT / f'code/data/splits/{task}_split.json').read_text())
            self.assertEqual(sorted(split['test']), sorted(load_panel(f'heldout/{task}')['case_ids']))


class PixelOverlapTests(DisplayedValueMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = ext.pixel_overlap_rows()

    def test_tables_c3_c4_points(self):
        # Task: same/release, cross-bundle, gap, FP-only gap, FN-only gap (Tables C.3(a)-(b)).
        expected = {'kvasir': ('0.842', '0.499', '0.343', '0.467', '0.319'),
                    'acdc_lv': ('0.860', '0.465', '0.394', '0.503', '0.365'),
                    'riga_cup': ('0.842', '0.486', '0.355', '0.447', '0.340'),
                    'riga_disc': ('0.840', '0.397', '0.443', '0.514', '0.459'),
                    'kvasir_grouped': (None, None, '0.390', None, None)}
        for task, (same, cross, gap, fp, fn) in expected.items():
            with self.subTest(task=task):
                err = by(self.rows, task=task, error='err')
                self.assertShown(err['gap'], gap)
                if same:
                    self.assertShown(err['same'], same)
                    self.assertShown(err['cross'], cross)
                    self.assertShown(by(self.rows, task=task, error='fp')['gap'], fp)
                    self.assertShown(by(self.rows, task=task, error='fn')['gap'], fn)
                self.assertEqual((err['n_both_empty_same'], err['n_both_empty_cross']), (0, 0))
        kvasir_fp = by(self.rows, task='kvasir', error='fp')
        self.assertEqual((kvasir_fp['n_both_empty_same'], kvasir_fp['n_comparisons_same'],
                          kvasir_fp['n_both_empty_cross'], kvasir_fp['n_comparisons_cross']), (905, 9840, 4524, 103680))
        self.assertShown(kvasir_fp['conditional_gap'], '0.465')

    def test_pairs_cover_the_floor_retained_heads(self):
        for task, panel_name in (('acdc_lv', 'primary/acdc_lv'), ('kvasir_grouped', 'kvasir_grouped/primary')):
            pixel, kept = load_pixel_overlap(task), floor_panel(load_panel(panel_name))
            self.assertEqual(list(zip(pixel['bundle'], pixel['seed'])), list(zip(kept['bundle'], kept['seed'])))
            self.assertTrue(np.array_equal(pixel['case_ids'], kept['case_ids']))
            self.assertEqual(pixel['err_dice'].dtype, np.float32)

    def test_loader_rejects_inconsistent_empty_masks(self):
        panel = load_pixel_overlap('riga_disc')
        with tempfile.TemporaryDirectory() as tmp:
            bad = {f'pixel_overlap/riga_disc/{k}': v.copy() for k, v in panel.items()}
            bad['pixel_overlap/riga_disc/err_dice'][0, 0] = np.nan  # empty error masks but defined FP/FN Dice
            np.savez(Path(tmp) / 'pixel_overlap.npz', **bad)
            entry = next(p for p in json.loads((ROOT / 'data/metadata.json').read_text())['panels']
                         if p['key'] == 'pixel_overlap/riga_disc')
            (Path(tmp) / 'metadata.json').write_text(json.dumps({'panels': [entry]}))
            with self.assertRaisesRegex(ValueError, 'empty'):
                load_pixel_overlap('riga_disc', tmp)


class ReleaseIntegrityTests(unittest.TestCase):
    def test_every_metadata_panel_loads(self):
        metadata = json.loads((ROOT / 'data/metadata.json').read_text())
        for entry in metadata['panels']:
            with self.subTest(panel=entry['key']):
                self.assertIn(entry['pack'], PACKS)
                if entry['key'].startswith('pixel_overlap/'):
                    panel = load_pixel_overlap(entry['key'].split('/', 1)[1])
                    self.assertEqual(3 * panel['err_dice'].size, entry['n_score_values'])
                    continue
                panel = (load_amos(entry['key'][5:]) if entry['key'].startswith('amos/')
                         else load_panel(entry['key']))
                self.assertEqual(panel['dice'].size, entry['n_score_values'])

    def test_manifest_lists_and_matches_all_data_and_reference_outputs(self):
        lines = (ROOT / 'MANIFEST.sha256').read_text().splitlines()
        listed = {}
        for line in lines:
            digest, path = line.split('  ', 1)
            self.assertRegex(digest, r'^[0-9a-f]{64}$')
            listed[path] = digest
        expected = sorted(str(p.relative_to(ROOT)) for folder in ('data', 'reference_results')
                          for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix != '.md')
        self.assertEqual(sorted(listed), expected)
        for path, digest in listed.items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)

    def test_croissant_files_match_the_manifest(self):
        croissant = json.loads((ROOT / 'croissant.json').read_text())
        manifest = dict(reversed(line.split('  ', 1)) for line in
                        (ROOT / 'MANIFEST.sha256').read_text().splitlines())
        files = [d for d in croissant['distribution'] if d['@type'] == 'cr:FileObject']
        self.assertTrue(files)
        for item in files:
            with self.subTest(file=item['@id']):
                self.assertEqual(item['contentUrl'], RAW + item['@id'])
                self.assertEqual(item['sha256'], manifest[item['@id']])


if __name__ == '__main__':
    unittest.main(verbosity=2)
