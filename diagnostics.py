"""Appendix diagnostics from the released score packs (one analysis per run).

bundle_dice        Table C.1 and Appendix G.2: mean Dice per bundle, frozen and fine-tuned, before the floor
failure_metrics    Table D.2(a) kappa, Yule's Q, disagreement and double fault at P33; Appendix D.2 text
failure_intervals  Table D.3 phi-gap and joint-fail-ratio intervals at P25/P33/P50 (also Table D.2)
calibration        Table D.1(c) calibration split and beta_same; patient-resampled intervals of Appendix D.1
item_difficulty    Table D.5 and Appendix D.4: residualised gaps, panels and both permutation nulls
tau_sweep          Table D.4 conditional mean-score threshold crossing
vit_diagnostic     Table F.1 intervals of the five-ViT diagnostic
checkpoints        Table F.2(a)-(b): checkpoint tiers, primary-pool decomposition and pair counts
unet_nine          Table G.1 nine-bundle rows (UNet-skip and frozen 1x1 on the same nine bundles)
pipeline_metrics   Table G.6 kappa, Q, disagreement and double fault for UNet-skip and nnU-Net
hierarchy          Appendix C.1 self-pair sensitivity and patient-aggregated gaps

--intervals adds the paper's bootstrap intervals and permutation nulls with the draw counts and
seeds recorded in the study (listed in each row); without it, only point estimates are computed.
"""
import argparse
import csv
import hashlib
import itertools
import random
from pathlib import Path
import sys

import numpy as np

from correlation import VITS, _pair_masks, _point, correlate, hierarchical_sensitivity
from effective_size import _interval, enumerate_pools
from scores import (PRIMARY_TASKS, correlation_matrix, expanded_panel, floor_panel, load_panel,
                    select_rows)
from sensitivity import _failure_stats, _quality_stats

ANALYSES = ('bundle_dice', 'failure_metrics', 'failure_intervals', 'calibration', 'item_difficulty',
            'tau_sweep', 'vit_diagnostic', 'checkpoints', 'unet_nine', 'pipeline_metrics', 'hierarchy')
CLUSTERED = ('acdc_lv', 'brats_wt')
METRICS = ('kappa', 'q', 'disagreement', 'double_fault')
TAUS = (0.70, 0.75, 0.80, 0.85, 0.90)
BROAD_FAMILY = {'dinov2_vitb14': 'dinov2', 'dinov2_vits14': 'dinov2', 'clip_vitb16': 'openai_clip',
                'clip_vitl14': 'openai_clip', 'resnet18': 'resnet', 'resnet50': 'resnet'}
NINE_BUNDLES_EXCLUDE = ('clip_vitb16', 'clip_vitl14')
CALIBRATION_FRACTION = 0.35


# ---- shared helpers -----------------------------------------------------------------------

def _draws(n, b, seed, labels=None):
    """fmpool.estimators._bootstrap_case_indices: cases, or whole sorted clusters."""
    groups = None if labels is None else [np.flatnonzero(labels == u) for u in sorted(set(labels))]
    rng = np.random.default_rng(seed)
    for _ in range(b):
        yield (rng.integers(0, n, n) if groups is None else
               np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]))


def _finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float('nan')


def _ci(row, name, values):
    row.update(_interval(values, name))
    return row


def _primary(task, data_dir):
    return floor_panel(load_panel(f'primary/{task}', data_dir))


def _labels(panel, task):
    return panel['cluster_ids'] if task in CLUSTERED else None


def _with_columns(panel, columns):
    return {k: (v[:, columns] if k in ('dice', 'prediction_empty') else
                v[columns] if k in ('case_ids', 'cluster_ids') else v) for k, v in panel.items()}


def _patient_means(panel):
    """Average every head's Dice within each patient (sorted), as compute_all._aggregate_by_unit."""
    units = sorted(set(panel['cluster_ids']))
    dice = np.stack([panel['dice'][:, panel['cluster_ids'] == u].mean(axis=1) for u in units], axis=1)
    return {**{k: v for k, v in panel.items() if k not in ('dice', 'case_ids', 'cluster_ids')},
            'dice': dice, 'case_ids': np.asarray(units), 'cluster_ids': np.asarray(units)}


def _metric_matrices(success, official=True):
    """Pairwise kappa, Yule's Q, disagreement and double fault (stats/analysis/failure_metrics.py).

    official=True scores kappa as 1 when two heads always agree (error-consistency convention)."""
    data = success.astype(float)
    total = float(data.shape[1])
    n11 = data @ data.T
    count = data.sum(axis=1)
    n10, n01 = count[:, None] - n11, count[None, :] - n11
    n00 = total - n11 - n01 - n10
    observed = (n11 + n00) / total
    p = data.mean(axis=1)
    expected = np.outer(p, p) + np.outer(1 - p, 1 - p)
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = (observed - expected) / (1 - expected)
        q = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10)
    if official:
        kappa = np.where(observed == 1.0, 1.0, kappa)
    return {'kappa': kappa, 'q': q, 'disagreement': (n01 + n10) / total, 'double_fault': n00 / total}


def _metric_gaps(success, i, j, same, official=True, keep=None):
    keep = np.ones(len(i), bool) if keep is None else keep
    out = {}
    for name, matrix in _metric_matrices(success, official).items():
        values = matrix[i, j]
        a, b = _finite_mean(values[same & keep]), _finite_mean(values[~same & keep])
        out[name] = (a, b, a - b)
    return out


def _supported(dice, threshold):
    failures = (dice < threshold).sum(axis=1)
    return failures, (failures >= 5) & (dice.shape[1] - failures >= 5)


# ---- Table C.1 ---------------------------------------------------------------------------------

def bundle_dice_rows(data_dir=None, intervals=False):
    sources = [(task, 'frozen', f'primary/{task}') for task in PRIMARY_TASKS]
    sources += [(task, f'ft_lr{rate}', f'ft/lr{rate}/{task}') for task in ('kvasir', 'riga_cup') for rate in ('1e-4', '1e-5')]
    rows = []
    for task, recipe, name in sources:
        panel = load_panel(name, data_dir)
        for bundle in np.unique(panel['bundle']):
            mean = float(panel['dice'][panel['bundle'] == bundle].mean())
            rows.append(dict(table='C.1' if recipe == 'frozen' else 'G.2 text', task=task, recipe=recipe,
                             bundle=bundle, mean_dice=mean, passes_floor=mean >= 0.30))
    return rows


# ---- Table D.2(a) and Appendix D.2 ----------------------------------------------------------

def failure_metrics_rows(data_dir=None, intervals=False):
    rows = []
    for index, task in enumerate(PRIMARY_TASKS):
        panel = _primary(task, data_dir)
        columns = np.argsort(panel['case_ids'], kind='stable')  # the study sorts case IDs here
        dice, labels = panel['dice'][:, columns], panel['cluster_ids'][columns]
        threshold = float(np.percentile(dice, 33))
        success = dice >= threshold
        i, j, same, *_ = _pair_masks(panel, 'release')
        names = np.array([f'{b}/seed_{s}' for b, s in zip(panel['bundle'], panel['seed'])])
        row = dict(table='D.2(a)', task=task, threshold=threshold, n_members=len(dice))
        for name, (a, b, gap) in _metric_gaps(success, i, j, same).items():
            row.update({f'{name}_same': a, f'{name}_cross': b, f'{name}_gap': gap})
        never = success.all(axis=1) | (~success).all(axis=1)
        keep = ~(never[i] | never[j])
        failures, ok = _supported(dice, threshold)
        row.update(n_never_failing=int(never.sum()), never_failing=';'.join(names[never]),
                   kappa_gap_without_never_failing=_metric_gaps(success, i, j, same, False, keep)['kappa'][2],
                   unsupported_heads=';'.join(f'{n}:{f}' for n, f, s in zip(names, failures, ok) if not s))
        if intervals:
            row.update(bootstrap=2000, bootstrap_seed=index * 10007, bootstrap_unit='case')
            draws = [_metric_gaps(success[:, idx], i, j, same)
                     for idx in _draws(dice.shape[1], 2000, index * 10007)]
            for name in METRICS:
                _ci(row, f'{name}_gap', [d[name][2] for d in draws])
            if task in CLUSTERED:
                patient = [_metric_gaps(success[:, idx], i, j, same, False)['kappa'][2]
                           for idx in _draws(dice.shape[1], 2000, 0, labels)]
                _ci(row, 'kappa_gap_patient', patient)
        rows.append(row)
    return rows


# ---- Table D.3 --------------------------------------------------------------------------------

def failure_intervals_rows(data_dir=None, intervals=False):
    """Percentile thresholds are re-estimated on every draw; ACDC/BraTS resample patients."""
    rows = []
    for index, task in enumerate(PRIMARY_TASKS):
        panel = _primary(task, data_dir)
        for percentile in (25, 33, 50):
            stats = _failure_stats(panel, percentile)
            row = dict(table='D.3', task=task, percentile=percentile, threshold=stats['threshold'],
                       same_fail_phi=stats['same_fail_phi'], cross_fail_phi=stats['cross_fail_phi'],
                       gap_fail_phi=stats['gap_fail_phi'], joint_fail_rate_ratio=stats['joint_fail_rate_ratio'],
                       n_same_defined_pairs=stats['n_same_defined_pairs'], n_same_pairs=stats['n_same_pairs'],
                       n_cross_defined_pairs=stats['n_cross_defined_pairs'], n_cross_pairs=stats['n_cross_pairs'])
            if intervals:
                row.update(bootstrap=1000, bootstrap_seed=index * 1009,
                           bootstrap_unit='patient' if task in CLUSTERED else 'case')
                draws = [_failure_stats(_with_columns(panel, idx), percentile)
                         for idx in _draws(panel['dice'].shape[1], 1000, index * 1009, _labels(panel, task))]
                _ci(row, 'gap_fail_phi', [d['gap_fail_phi'] for d in draws])
                _ci(row, 'joint_fail_rate_ratio', [d['joint_fail_rate_ratio'] for d in draws])
            rows.append(row)
    return rows


# ---- Table D.1(c) and Appendix D.1 --------------------------------------------------------------

def _calibration_split(task, panel):
    """compute_all._stable_unit_split: 35% of units in SHA-256 order of 'task:unit'."""
    units = sorted(set(panel['cluster_ids']))
    keyed = sorted((hashlib.sha256(f'{task}:{u}'.encode()).hexdigest(), u) for u in units)
    n_cal = max(1, min(len(units) - 1, int(round(len(units) * CALIBRATION_FRACTION))))
    return np.isin(panel['cluster_ids'], [u for _, u in keyed[:n_cal]]), n_cal, len(units) - n_cal


def calibration_rows(data_dir=None, intervals=False):
    rows = []
    for index, task in enumerate(PRIMARY_TASKS):
        full = load_panel(f'primary/{task}', data_dir)
        cal, n_cal, n_eval = _calibration_split(task, full)
        means = {b: full['dice'][full['bundle'] == b][:, cal].mean(axis=1).mean() for b in set(full['bundle'])}
        selected = sorted(b for b, m in means.items() if m >= 0.30)
        evaluation = _with_columns(select_rows(full, np.isin(full['bundle'], selected)), ~cal)
        split = correlate(evaluation, unit='case')
        primary = floor_panel(full)
        quality = _quality_stats(primary)
        row = dict(table='D.1(c)', task=task, n_calibration_units=n_cal, n_evaluation_units=n_eval,
                   selected_K=len(selected), calibration_gap=split['gap'],
                   beta_same=quality['beta_same'], r_squared=quality['r_squared'])
        if intervals:
            n = evaluation['dice'].shape[1]
            masks = _pair_masks(evaluation, 'release')
            row.update(calibration_bootstrap=2000, calibration_seed=0, beta_bootstrap=1000,
                       beta_seed=index * 1009)
            _ci(row, 'calibration_gap', [_point(evaluation['dice'][:, idx], masks)[2]
                                         for idx in _draws(n, 2000, 0)])
            _ci(row, 'beta_same', [_quality_stats(_with_columns(primary, idx))['beta_same']
                                   for idx in _draws(primary['dice'].shape[1], 1000, index * 1009)])
            if task in CLUSTERED:
                _ci(row, 'calibration_gap_patient', [_point(evaluation['dice'][:, idx], masks)[2]
                                                     for idx in _draws(n, 2000, 0, evaluation['cluster_ids'])])
                _ci(row, 'beta_same_patient', [
                    _quality_stats(_with_columns(primary, idx))['beta_same']
                    for idx in _draws(primary['dice'].shape[1], 1000, index * 1009, primary['cluster_ids'])])
        rows.append(row)
    return rows


# ---- Table D.5 and Appendix D.4 -------------------------------------------------------------------

def _panel_setup(bundles):
    """All 4-of-8 reference-family subsets; the complementary families form the evaluation panel."""
    families = sorted({BROAD_FAMILY.get(b, b) for b in bundles})
    members = {f: [k for k, b in enumerate(bundles) if BROAD_FAMILY.get(b, b) == f] for f in families}
    panels = []
    for reference in itertools.combinations(families, len(families) // 2):
        evaluation = np.array([k for k, b in enumerate(bundles) if BROAD_FAMILY.get(b, b) not in reference])
        labels = np.asarray(bundles)[evaluation]
        a, b = np.triu_indices(len(evaluation), 1)
        dino = np.isin(labels, ['dinov2_vitb14', 'dinov2_vits14'])
        same = (labels[a] == labels[b]) | (dino[a] & dino[b])
        panels.append((reference, evaluation, a, b, same))
    return families, members, panels


def _to_corr(cov):
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        r = cov / np.outer(sd, sd)
    r[sd < 1e-8, :] = np.nan
    r[:, sd < 1e-8] = np.nan
    return r


def _panel_gaps(dice, setup):
    """Raw and residualised gaps of every panel. OLS on one common regressor (the equal-weight mean of
    the reference families' mean traces) has the closed-form residual covariance C - c c' / v."""
    families, members, panels = setup
    stacked = np.vstack([dice] + [dice[members[f]].mean(axis=0, keepdims=True) for f in families])
    centred = stacked - stacked.mean(axis=1, keepdims=True)
    cov = centred @ centred.T / stacked.shape[1]
    position = {f: len(dice) + k for k, f in enumerate(families)}
    out = []
    for reference, evaluation, a, b, same in panels:
        ref = [position[f] for f in reference]
        cee = cov[np.ix_(evaluation, evaluation)]
        v = cov[np.ix_(ref, ref)].mean()
        c = cov[np.ix_(evaluation, ref)].mean(axis=1)
        residual = cee - np.outer(c, c) / v if np.sqrt(v) >= 1e-12 else cee
        gaps = []
        for matrix in (cee, residual):
            r = _to_corr(matrix)[a, b]
            gaps.append(_finite_mean(r[same]) - _finite_mean(r[~same]))
        out.append(gaps)
    return np.asarray(out)


def _relabelled(bundles, rng):
    """compute_audit_sensitivity._shuffle_family_labels: permute labels over sorted (bundle, seed) keys."""
    labels = list(bundles)
    rng.shuffle(labels)
    occurrence, keys = {}, []
    for label in labels:
        occurrence[label] = occurrence.get(label, 0) + 1
        keys.append((label, occurrence[label]))
    return keys


def _raw_null(panel, observed):
    r = correlation_matrix(panel['dice'])
    i, j = np.triu_indices(len(r), 1)
    values, rng, null = r[i, j], random.Random(0), []
    for _ in range(1000):
        labels = np.array([k[0] for k in _relabelled(panel['bundle'], rng)])
        dino = np.isin(labels, ['dinov2_vitb14', 'dinov2_vits14'])
        same = (labels[i] == labels[j]) | (dino[i] & dino[j])
        null.append(_finite_mean(values[same]) - _finite_mean(values[~same]))
    null = np.asarray(null)
    return dict(raw_null_n_perm=len(null), raw_null_q025=float(np.quantile(null, 0.025)),
                raw_null_q975=float(np.quantile(null, 0.975)),
                raw_null_f_perm=float((1 + np.sum(null >= observed)) / (len(null) + 1)))


def _residual_null(panel):
    rng, null = random.Random(0), []
    for _ in range(200):
        keys = _relabelled(panel['bundle'], rng)
        order = sorted(range(len(keys)), key=lambda k: keys[k])
        null.append(_panel_gaps(panel['dice'][order], _panel_setup([keys[k][0] for k in order])).mean(axis=0))
    null = np.asarray(null)
    return dict(residual_null_n_perm=len(null), residual_null_mean=float(null[:, 1].mean()),
                residual_null_max=float(null[:, 1].max()))


def item_difficulty_rows(data_dir=None, intervals=False):
    rows = []
    for task in PRIMARY_TASKS:
        panel = _primary(task, data_dir)
        setup = _panel_setup(list(panel['bundle']))
        gaps = _panel_gaps(panel['dice'], setup)
        raw, residual = gaps.mean(axis=0)
        eligible = gaps[np.isfinite(gaps).all(axis=1), 1]
        row = dict(table='D.5', task=task, raw_gap=raw, residual_gap=residual, residual_minus_raw=residual - raw,
                   residual_over_raw=residual / raw, n_panels=len(gaps), n_eligible=len(eligible),
                   n_positive=int((eligible > 1e-12).sum()), residual_panel_min=float(eligible.min()),
                   residual_panel_max=float(eligible.max()), residual_panel_sd=float(eligible.std()))
        if task in CLUSTERED:
            patient = _patient_means(panel)
            row['patient_residual_minus_raw'] = float(np.diff(_panel_gaps(patient['dice'], setup).mean(axis=0))[0])
        if intervals:
            draws = np.array([_panel_gaps(panel['dice'][:, idx], setup).mean(axis=0) for idx in
                              _draws(panel['dice'].shape[1], 2000, 0, _labels(panel, task))])
            both = np.isfinite(draws).all(axis=1)
            ratio = np.divide(draws[:, 1], draws[:, 0], out=np.full(len(draws), np.nan),
                              where=both & (np.abs(draws[:, 0]) >= 1e-12))
            row.update(bootstrap=2000, bootstrap_seed=0, bootstrap_unit='patient' if task in CLUSTERED else 'case')
            for name, values in (('raw_gap', draws[:, 0]), ('residual_gap', draws[:, 1]),
                                 ('residual_minus_raw', (draws[:, 1] - draws[:, 0])[both]),
                                 ('residual_over_raw', ratio)):
                _ci(row, name, values)
            row.update(_raw_null(panel, correlate(panel)['gap']), **_residual_null(panel))
        rows.append(row)
    return rows


# ---- Table D.4 ----------------------------------------------------------------------------------

def _recovery(pools, dice, tau):
    """Mean over pools of P[pool-mean Dice >= tau | some member < tau]; undefined pools omitted."""
    rates = []
    for start in range(0, len(pools), 1024):
        data = dice[pools[start:start + 1024]]
        fails = (data < tau).any(axis=1)
        passes = data.mean(axis=1) >= tau
        denominator = fails.sum(axis=1)
        rates.append(np.where(denominator > 0, (fails & passes).sum(axis=1) / np.maximum(denominator, 1), np.nan))
    return float(np.nanmean(np.concatenate(rates)))


def tau_sweep_rows(data_dir=None, intervals=False):
    rows = []
    for task in PRIMARY_TASKS:
        panel = _primary(task, data_dir)
        same = np.array([np.flatnonzero(panel['bundle'] == b) for b in np.unique(panel['bundle'])])
        diverse = enumerate_pools(panel)
        for tau in TAUS:
            a, b = _recovery(same, panel['dice'], tau), _recovery(diverse, panel['dice'], tau)
            rows.append(dict(table='D.4', task=task, tau=tau, same_bundle_rate=a, diverse_rate=b,
                             recovery_advantage=b - a, n_same_bundle_pools=len(same), n_diverse_pools=len(diverse)))
    return rows


# ---- Table F.1 ---------------------------------------------------------------------------------

def vit_diagnostic_rows(data_dir=None, intervals=False):
    rows = []
    for task in PRIMARY_TASKS:
        primary = _primary(task, data_dir)
        panel = select_rows(primary, np.isin(primary['bundle'], VITS))
        point = correlate(panel, 'checkpoint')
        row = dict(table='F.1', task=task, K=point['retained_K'], K_available=len(VITS),
                   seed_correlation=point['within'], cross_pretraining=point['cross'],
                   cross_bundle_reference=correlate(primary)['cross'], vit_gap=point['gap'],
                   n_seed_pairs=point['n_within_pairs'], n_cross_pretraining_pairs=point['n_cross_pairs'])
        if intervals:
            masks = _pair_masks(panel, 'checkpoint')
            row.update(bootstrap=2000, bootstrap_seed=0)
            draws = np.array([_point(panel['dice'][:, idx], masks)[:3]
                              for idx in _draws(panel['dice'].shape[1], 2000, 0)])
            for column, name in enumerate(('seed_correlation', 'cross_pretraining', 'vit_gap')):
                _ci(row, name, draws[:, column])
            if task in CLUSTERED:
                _ci(row, 'vit_gap_patient', [_point(panel['dice'][:, idx], masks)[2] for idx in
                                             _draws(panel['dice'].shape[1], 2000, 0, panel['cluster_ids'])])
        rows.append(row)
    return rows


# ---- Table F.2 ---------------------------------------------------------------------------------

def _mask_gap(panel, within, cross):
    r = correlation_matrix(panel['dice'])
    i, j = np.triu_indices(len(r), 1)
    values = r[i, j]
    return _finite_mean(values[within]) - _finite_mean(values[cross])


def checkpoints_rows(data_dir=None, intervals=False):
    rows = []
    for task in PRIMARY_TASKS:
        expanded, primary = floor_panel(expanded_panel(task, data_dir)), _primary(task, data_dir)
        tiers = correlate(expanded, 'family', bootstrap=2000 if intervals else 0)
        eleven = correlate(primary, 'family', bootstrap=2000 if intervals else 0)
        i, j, release, _, same_bundle, _ = _pair_masks(primary, 'release')
        dino = release & ~same_bundle
        row = dict(table='F.2', task=task, K=tiers['retained_K'], K_available=17,
                   same_bundle=tiers['same_checkpoint'],
                   n_same_bundle_pairs=int(_pair_masks(expanded, 'family')[4].sum()),
                   same_family=tiers['within'], n_same_family_pairs=tiers['n_within_pairs'],
                   cross_family=tiers['cross'], n_cross_family_pairs=tiers['n_cross_pairs'],
                   ckpt_gap=tiers['gap'], primary_ckpt_gap=eleven['gap'],
                   gap_same_bundle_only=_mask_gap(primary, same_bundle, ~release),
                   gap_dinov2_as_cross=correlate(primary, 'checkpoint')['gap'],
                   primary_distinct_ckpt=eleven['within'], n_primary_distinct_ckpt_pairs=eleven['n_within_pairs'],
                   cross_excluding_family=eleven['cross'],
                   same_bundle_gap_vs_cross_excluding_family=eleven['same_checkpoint'] - eleven['cross'],
                   n_same_bundle_seed_pairs=int(same_bundle.sum()), n_release_pairs=int(release.sum()),
                   n_dinov2_release_pairs=int(dino.sum()))
        if intervals:
            row.update(bootstrap=2000, bootstrap_seed=0, bootstrap_unit=tiers['ci_unit'],
                       ckpt_gap_ci95_low=tiers['gap_ci_low'], ckpt_gap_ci95_high=tiers['gap_ci_high'],
                       primary_ckpt_gap_ci95_low=eleven['gap_ci_low'], primary_ckpt_gap_ci95_high=eleven['gap_ci_high'])
        rows.append(row)
    return rows


# ---- Table G.1 ---------------------------------------------------------------------------------

def unet_nine_rows(data_dir=None, intervals=False):
    rows = []
    for task in ('kvasir', 'acdc_lv', 'riga_cup', 'riga_disc'):
        row = dict(table='G.1', task=task, source_bundles=9)
        for prefix, name in (('unet', f'unet_skip/{task}'), ('frozen', f'primary/{task}')):
            panel = load_panel(name, data_dir)
            panel = floor_panel(select_rows(panel, ~np.isin(panel['bundle'], NINE_BUNDLES_EXCLUDE)))
            r = correlate(panel, bootstrap=2000 if intervals else 0)
            row.update({f'{prefix}_K': r['retained_K'], f'{prefix}_within': r['within'],
                        f'{prefix}_cross': r['cross'], f'{prefix}_gap': r['gap']})
            if intervals:
                row.update({f'{prefix}_gap_ci95_low': r['gap_ci_low'], f'{prefix}_gap_ci95_high': r['gap_ci_high'],
                            'bootstrap': 2000, 'bootstrap_seed': 0, 'bootstrap_unit': r['ci_unit']})
        rows.append(row)
    return rows


# ---- Table G.6 ---------------------------------------------------------------------------------

def pipeline_metrics_rows(data_dir=None, intervals=False):
    """Pooled P33 fixed; draws without five failures and passes per member are skipped."""
    rows = []
    for name, task, grouping in (('unet_skip', 'riga_cup', 'release'), ('unet_skip', 'riga_disc', 'release'),
                                 ('nnunet', 'riga_cup', 'fold')):
        panel = load_panel(f'nnunet/riga_cup_2d' if name == 'nnunet' else f'{name}/{task}', data_dir)
        panel = panel if name == 'nnunet' else floor_panel(panel)
        dice = panel['dice']
        threshold = float(np.percentile(dice, 33))
        i, j, same, *_ = _pair_masks(panel, grouping)
        row = dict(table='G.6', pipeline=name, task=task, n_members=len(dice), threshold=threshold,
                   all_members_supported=bool(_supported(dice, threshold)[1].all()))
        for metric, (a, b, gap) in _metric_gaps(dice >= threshold, i, j, same).items():
            row.update({f'{metric}_same': a, f'{metric}_cross': b, f'{metric}_gap': gap})
        if intervals:
            draws = []
            for idx in _draws(dice.shape[1], 2000, 0):
                sample = dice[:, idx]
                if _supported(sample, threshold)[1].all():
                    draws.append(_metric_gaps(sample >= threshold, i, j, same))
            row.update(bootstrap=2000, bootstrap_seed=0, n_supported_draws=len(draws))
            for metric in METRICS:
                _ci(row, f'{metric}_gap', [d[metric][2] for d in draws])
        rows.append(row)
    return rows


# ---- Appendix C.1 --------------------------------------------------------------------------------

def hierarchy_rows(data_dir=None, intervals=False):
    rows = []
    for task in PRIMARY_TASKS:
        primary = _primary(task, data_dir)
        row = dict(table='C.1', task=task)
        if task in CLUSTERED:
            for prefix, panel, grouping in (('patient', primary, 'release'),
                                            ('patient_ckpt', floor_panel(expanded_panel(task, data_dir)), 'family')):
                r = correlate(_patient_means(panel), grouping, bootstrap=2000 if intervals else 0)
                row.update({f'{prefix}_n_units': r['n_cases'], f'{prefix}_gap': r['gap']})
                if intervals:
                    row.update({f'{prefix}_gap_ci95_low': r['gap_ci_low'], f'{prefix}_gap_ci95_high': r['gap_ci_high']})
        if intervals:
            row.update(bootstrap=2000, bootstrap_seed=0)
            for suffix, drop in (('with_self_pairs', False), ('without_self_pairs', True)):
                h = hierarchical_sensitivity(primary, bootstrap=2000, drop_self=drop)
                row.update({f'hierarchical_{suffix}_low': h['hierarchical_gap_low'],
                            f'hierarchical_{suffix}_high': h['hierarchical_gap_high']})
        rows.append(row)
    return rows


RUNNERS = dict(zip(ANALYSES, (bundle_dice_rows, failure_metrics_rows, failure_intervals_rows, calibration_rows,
                              item_difficulty_rows, tau_sweep_rows, vit_diagnostic_rows, checkpoints_rows,
                              unet_nine_rows, pipeline_metrics_rows, hierarchy_rows)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--analysis', choices=ANALYSES, required=True)
    parser.add_argument('--intervals', action='store_true',
                        help="add the paper's bootstrap intervals and permutation nulls (study draw counts and seeds)")
    parser.add_argument('--data-dir', type=Path,
                        help='directory containing metadata.json and the NPZ packs (default: data/)')
    parser.add_argument('--output', type=Path, help='CSV destination (default: stdout)')
    args = parser.parse_args(argv)
    rows = RUNNERS[args.analysis](args.data_dir, args.intervals)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    stream = args.output.open('w', newline='') if args.output else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            stream.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
