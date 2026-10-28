"""Appendix extensions from data/extensions.npz (one analysis per run).

mask_align        Table B.2: default nearest versus nearest-exact mask resizing, paired difference
per_rater         Table C.2: RIGA heads re-scored against each rater and the 4-of-6 consensus
kvasir_grouped    Table C.5 and the content-group Kvasir rows of Tables G.2 and G.3
medical_encoders  Table H.2: RETFound, RAD-DINO and BiomedParse against the primary pool
amos              Tables G.7 and G.8: native-3D AMOS22 comparison
heldout           Table I.1(a): frozen 1x1 rows of the primary and four held-out tasks
pixel_overlap     Tables C.3 and C.4: error-mask overlap of head pairs (data/pixel_overlap.npz)

Phikon outputs are not released, so the Phikon rows and every value that pools Phikon
with other encoders (pooled medical rows, mutual rho, the 15-bundle pool) are not computed.
--bootstrap 2000 reproduces the paper's intervals with its seeds; the default 0 gives points.
"""
import argparse
import csv
import random
from pathlib import Path
import sys
import warnings

import numpy as np

from control_comparisons import paired_gap
from correlation import _pair_masks, _point, correlate
from effective_size import _interval, _pool_summary
from scores import (PRIMARY_TASKS, correlation_matrix, expanded_panel, floor_panel, load_amos,
                    load_panel, load_pixel_overlap, select_rows)

ANALYSES = ('mask_align', 'per_rater', 'kvasir_grouped', 'medical_encoders', 'amos', 'heldout', 'pixel_overlap')
MASK_TASKS = ('kvasir', 'acdc_lv', 'riga_cup', 'riga_disc')
MEDICAL_TASKS = ('kvasir', 'riga_cup', 'riga_disc')
HELDOUT_TASKS = (*PRIMARY_TASKS, 'isic2018', 'msd_hippocampus', 'msd_spleen', 'msd_heart')
PIXEL_TASKS = ('kvasir', 'acdc_lv', 'riga_cup', 'riga_disc', 'kvasir_grouped')
COLLAPSED_HEAD = ('clip_vitl14', 42)  # content-group Kvasir, encoder lr 1e-4 (paper Appendix G.2)
AMOS_PEARSON_SEED = 20260726  # Python random.Random, consumed by CT and then MRI
AMOS_SAME, AMOS_CROSS = [(0, 1)], [(0, 2), (1, 2)]  # MRSegmentator fold pair; fold x TotalSegmentator


def _case_draws(panel, bootstrap, seed=0):
    """Index draws of correlation.correlate: whole patients where present, else cases."""
    n = panel['dice'].shape[1]
    labels = sorted(set(panel['cluster_ids']))
    groups = ([np.flatnonzero(panel['cluster_ids'] == label) for label in labels]
              if len(labels) < n else None)
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap):
        yield (rng.integers(0, n, n) if groups is None else
               np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]))


def _with_interval(row, name, draws, bootstrap):
    if bootstrap:
        row.update(_interval(draws, name))
    return row


def mask_align_rows(data_dir=None, bootstrap=0):
    """Table B.2: both conventions share every case draw (patients for ACDC LV)."""
    rows = []
    for task in MASK_TASKS:
        nearest = floor_panel(load_panel(f'primary/{task}', data_dir))
        exact = floor_panel(load_panel(f'mask_align/{task}', data_dir))
        for key in ('case_ids', 'cluster_ids'):
            if not np.array_equal(nearest[key], exact[key]):
                raise ValueError(f'{task}: mask conventions are not case-aligned')
        masks = [_pair_masks(p, 'release') for p in (nearest, exact)]
        draws = np.array([[_point(p['dice'][:, idx], m)[2] for p, m in zip((nearest, exact), masks)]
                          for idx in _case_draws(nearest, bootstrap)]).reshape(-1, 2)
        a, b = correlate(nearest), correlate(exact)
        row = dict(table='B.2', task=task, n_cases=a['n_cases'], n_units=a['n_units'],
                   K_nearest=a['retained_K'], K_exact=b['retained_K'],
                   within_nearest=a['within'], cross_nearest=a['cross'], gap_nearest=a['gap'],
                   within_exact=b['within'], cross_exact=b['cross'], gap_exact=b['gap'],
                   gap_difference=b['gap'] - a['gap'], bootstrap=bootstrap, bootstrap_seed=0,
                   bootstrap_unit='patient' if a['n_units'] < a['n_cases'] else 'image')
        _with_interval(row, 'gap_nearest', draws[:, 0], bootstrap)
        _with_interval(row, 'gap_exact', draws[:, 1], bootstrap)
        both = np.isfinite(draws).all(axis=1)
        rows.append(_with_interval(row, 'gap_difference', (draws[:, 1] - draws[:, 0])[both], bootstrap))
    return rows


def _correlation_row(panel, bootstrap, grouping='release'):
    r = correlate(panel, grouping, bootstrap=bootstrap)
    row = dict(K=r['retained_K'], n_members=r['n_members'], within=r['within'], cross=r['cross'],
               gap=r['gap'])
    if bootstrap:
        row.update(gap_ci95_low=r['gap_ci_low'], gap_ci95_high=r['gap_ci_high'],
                   gap_n_valid=r['n_valid_boot'])
    return row


def per_rater_rows(data_dir=None, bootstrap=0):
    """Table C.2: GT-specific floors and a pool fixed by the consensus floor."""
    rows = []
    for table, task in (('C.2(a)', 'riga_cup'), ('C.2(b)', 'riga_disc')):
        consensus = load_panel(f'primary/{task}', data_dir)  # the same heads, consensus reference
        kept = sorted(set(floor_panel(consensus)['bundle']))
        references = [(f'rater_{k}', load_panel(f'per_rater/rater_{k}/{task}', data_dir))
                      for k in range(1, 7)] + [('consensus', consensus)]
        for pool in ('gt_specific', 'fixed_consensus'):
            block = []
            for reference, panel in references:
                chosen = (floor_panel(panel) if pool == 'gt_specific'
                          else select_rows(panel, np.isin(panel['bundle'], kept)))
                excluded = ';'.join(sorted(set(panel['bundle']) - set(chosen['bundle'])))
                block.append(dict(table=table, task=task, pool=pool, reference=reference,
                                  excluded_bundles=excluded, **_correlation_row(chosen, bootstrap),
                                  bootstrap=bootstrap, bootstrap_seed=0, bootstrap_unit='image'))
            single = [row['gap'] for row in block[:-1]]
            block[-1].update(single_rater_gap_min=min(single), single_rater_gap_max=max(single),
                             consensus_minus_single_rater_max=block[-1]['gap'] - max(single))
            rows.extend(block)
    return rows


def _grouped_c5_rows(data_dir, bootstrap):
    official, grouped = (load_panel(n, data_dir) for n in ('primary/kvasir', 'kvasir_grouped/primary'))
    shared = len(set(official['case_ids']) & set(grouped['case_ids']))
    sources = {'official': (official, expanded_panel('kvasir', data_dir),
                            load_panel('unet_skip/kvasir', data_dir)),
               'content_group': (grouped, expanded_panel('kvasir', data_dir, grouped=True),
                                 load_panel('kvasir_grouped/unet_skip', data_dir))}
    rows = []
    for split, (primary, expanded, unet) in sources.items():
        for analysis, panel, grouping in (('primary', primary, 'release'),
                                          ('distinct_checkpoint', expanded, 'family'),
                                          ('unet_skip', unet, 'release')):
            rows.append(dict(table='C.5', split=split, analysis=analysis,
                             **_correlation_row(floor_panel(panel), bootstrap, grouping),
                             n_cases=panel['dice'].shape[1], n_test_images_shared_by_splits=shared,
                             bootstrap=bootstrap, bootstrap_seed=0, bootstrap_unit='image'))
    return rows


def _grouped_ft_rows(data_dir, bootstrap):
    frozen = load_panel('kvasir_grouped/primary', data_dir)
    rows = []
    for rate in ('1e-4', '1e-5'):
        ft = load_panel(f'kvasir_grouped/ft/lr{rate}', data_dir)
        point = correlate(floor_panel(ft))
        paired = paired_gap(ft, frozen, bootstrap)  # FT columns renamed: ft_gap -> gap, K_ft -> K
        rows.append(dict(table='G.2/G.3', split='content_group', analysis='fine_tuning',
                         recipe=f'ft_lr{rate}', n_members=point['n_members'],
                         within=point['within'], cross=point['cross'],
                         **{k.replace('ft_gap', 'gap').replace('K_ft', 'K'): v
                            for k, v in paired.items()}))
        if rate == '1e-4':
            member = (ft['bundle'] == COLLAPSED_HEAD[0]) & (ft['seed'] == COLLAPSED_HEAD[1])
            if member.sum() != 1 or ft['dice'][member].mean() >= 0.05:
                raise ValueError('expected one collapsed head with near-zero Dice')
            kept = floor_panel(select_rows(ft, ~member))
            rows.append(dict(table='G.2 text', split='content_group',
                             analysis='fine_tuning_excluding_collapsed_head', recipe='ft_lr1e-4',
                             excluded_member='{}/seed_{}'.format(*COLLAPSED_HEAD),
                             **_correlation_row(kept, bootstrap), bootstrap=bootstrap,
                             bootstrap_seed=0, bootstrap_unit='image'))
    return rows


def _bundle_dice_rows(data_dir):
    """Appendix G.2 text: per-bundle mean Dice, fine-tuned versus frozen, on both Kvasir splits."""
    names = {'official': ('primary/kvasir', 'ft/lr{}/kvasir'),
             'content_group': ('kvasir_grouped/primary', 'kvasir_grouped/ft/lr{}')}
    rows = []
    for split, (frozen_name, ft_name) in names.items():
        frozen = load_panel(frozen_name, data_dir)
        for rate in ('1e-4', '1e-5'):
            ft = load_panel(ft_name.format(rate), data_dir)
            for bundle in sorted(set(frozen['bundle'])):
                a = float(frozen['dice'][frozen['bundle'] == bundle].mean())
                b = float(ft['dice'][ft['bundle'] == bundle].mean())
                rows.append(dict(table='G.2 text', split=split, analysis='bundle_dice',
                                 recipe=f'ft_lr{rate}', bundle=bundle, frozen_mean_dice=a,
                                 ft_mean_dice=b, ft_minus_frozen=b - a))
    return rows


def kvasir_grouped_rows(data_dir=None, bootstrap=0):
    """Table C.5 (both splits) and the content-group rows of Tables G.2-G.3."""
    return (_grouped_c5_rows(data_dir, bootstrap) + _grouped_ft_rows(data_dir, bootstrap)
            + _bundle_dice_rows(data_dir))


def _medical_point(primary, medical, columns, masks):
    """Primary cross-bundle mean r, each encoder's mean r with all retained primary heads,
    and the full correlation matrix (primary heads first)."""
    r = correlation_matrix(np.vstack([primary['dice'], medical['dice']])[:, columns])
    m = len(primary['dice'])
    rho = {bundle: float(r[m + np.flatnonzero(medical['bundle'] == bundle)][:, :m].mean())
           for bundle in sorted(set(medical['bundle']))}
    return _point(primary['dice'][:, columns], masks)[1], rho, r


def medical_encoders_rows(data_dir=None, bootstrap=0):
    """Table H.2 rows for the released encoders; Phikon and pooled rows are not computed."""
    rows = []
    for task in MEDICAL_TASKS:
        primary = floor_panel(load_panel(f'primary/{task}', data_dir))
        medical = load_panel(f'medical/{task}', data_dir)
        if not np.array_equal(primary['case_ids'], medical['case_ids']):
            raise ValueError(f'{task}: medical and primary cases differ')
        masks = _pair_masks(primary, 'release')
        everything = np.arange(primary['dice'].shape[1])
        cross, rho, r = _medical_point(primary, medical, everything, masks)
        draws = [_medical_point(primary, medical, idx, masks)[:2]
                 for idx in _case_draws(primary, bootstrap)]
        m = len(primary['dice'])
        for encoder in sorted(set(medical['bundle'])):
            heads = m + np.flatnonzero(medical['bundle'] == encoder)
            by_bundle = {b: float(r[heads][:, np.flatnonzero(primary['bundle'] == b)].mean())
                         for b in sorted(set(primary['bundle']))}
            top = max(by_bundle, key=by_bundle.get)
            mean_dice = float(medical['dice'][medical['bundle'] == encoder].mean())
            row = dict(table='H.2', task=task, encoder=encoder, mean_dice=mean_dice,
                       passes_floor=mean_dice >= 0.30, rho_to_primary=rho[encoder],
                       cross=cross, delta_med=rho[encoder] - cross,
                       n_primary_heads=m, K_primary=len(set(primary['bundle'])),
                       top_primary_bundle=top, top_primary_rho=by_bundle[top],
                       bootstrap=bootstrap, bootstrap_seed=0, bootstrap_unit='image')
            _with_interval(row, 'rho_to_primary', [d[1][encoder] for d in draws], bootstrap)
            rows.append(_with_interval(row, 'delta_med', [d[1][encoder] - d[0] for d in draws],
                                       bootstrap))
    return rows


def _amos_pearson(dice):
    r = np.corrcoef(dice)
    same, cross = r[0, 1], (r[0, 2] + r[1, 2]) / 2
    return np.array([same, cross, same - cross])


def _failure_gaps(success):
    """Same-minus-cross kappa, Q, disagreement and double fault (study failure_metrics)."""
    data = success.astype(float)
    total = float(data.shape[1])
    n11 = data @ data.T
    count = data.sum(axis=1)
    n10, n01 = count[:, None] - n11, count[None, :] - n11
    n00 = total - n11 - n01 - n10
    p = data.mean(axis=1)
    expected = np.outer(p, p) + np.outer(1 - p, 1 - p)
    with np.errstate(divide='ignore', invalid='ignore'):
        metrics = {'kappa': ((n11 + n00) / total - expected) / (1 - expected),
                   'q': (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10),
                   'disagreement': (n01 + n10) / total, 'double_fault': n00 / total}
    gaps = {}
    for name, matrix in metrics.items():
        same, cross = ([matrix[i, j] for i, j in pairs] for pairs in (AMOS_SAME, AMOS_CROSS))
        same, cross = (np.asarray(v)[np.isfinite(v)] for v in (same, cross))
        gaps[name] = same.mean() - cross.mean() if len(same) and len(cross) else np.nan
    return gaps


def amos_rows(data_dir=None, bootstrap=0):
    """Tables G.7-G.8. Pearson draws follow the study's stdlib stream (CT, then MRI);
    effective size and failure metrics share numpy seed-0 volume draws, threshold fixed."""
    rng, rows = random.Random(AMOS_PEARSON_SEED), []
    for modality in ('ct', 'mri'):
        panel = load_amos(modality, data_dir)
        dice, n = panel['dice'], panel['dice'].shape[1]
        finite = np.isfinite(panel['organ_dice'][0]).sum(axis=1)
        threshold = float(np.percentile(dice, 33))
        pools = np.arange(3)[None, :]
        failures = (dice < threshold).sum(axis=1)
        row = dict(table='G.7/G.8', modality=modality, n_cases=n, n_organs=len(panel['organs']),
                   undefined_organ_cells_per_model=int((~np.isfinite(panel['organ_dice'][0])).sum()),
                   min_finite_organs_per_case=int(finite.min()),
                   max_finite_organs_per_case=int(finite.max()),
                   **dict(zip(('same_r', 'cross_r', 'gap'), _amos_pearson(dice))),
                   pooled_P33=threshold, M_eff=_pool_summary(dice, threshold, pools)['M_eff'],
                   member_failures=';'.join(map(str, failures)),
                   member_passes=';'.join(map(str, n - failures)),
                   **{f'{k}_gap': v for k, v in _failure_gaps(dice >= threshold).items()},
                   bootstrap=bootstrap, pearson_bootstrap_seed=AMOS_PEARSON_SEED,
                   volume_bootstrap_seed=0, bootstrap_unit='volume')
        if bootstrap:
            pearson = np.array([_amos_pearson(dice[:, [rng.randrange(n) for _ in range(n)]])
                                for _ in range(bootstrap)])
            for column, name in enumerate(('same_r', 'cross_r', 'gap')):
                row.update(_interval(pearson[:, column], name))
            volume = np.random.default_rng(0)
            meff, metrics = [], []
            for _ in range(bootstrap):
                sample = dice[:, volume.integers(0, n, size=n)]
                summary = _pool_summary(sample, threshold, pools)
                meff.append(summary['M_eff'])
                if summary['n_supported']:  # every model has >= 5 failures and >= 5 passes
                    metrics.append(_failure_gaps(sample >= threshold))
            row.update(_interval(meff, 'M_eff'))
            for name in ('kappa', 'q', 'disagreement', 'double_fault'):
                row.update(_interval([m[name] for m in metrics], f'{name}_gap'))
        rows.append(row)
    return rows


def _empty_mask_counts(panel):
    """Appendix I.1 text on MSD Spleen, from the recorded empty-target and empty-prediction flags."""
    target, prediction, dice = panel['target_empty'], panel['prediction_empty'], panel['dice']
    return dict(n_target_empty=int(target.sum()), n_predictions=int(prediction.size),
                n_nonempty_predictions=int((~prediction).sum()),
                n_heads_without_foreground=int(prediction.all(axis=1).sum()),
                n_empty_target_pairs=int(dice[:, target].size),
                n_empty_target_dice_one=int((dice[:, target] == 1).sum()),
                n_nonempty_target_pairs=int(dice[:, ~target].size),
                n_nonempty_target_dice_zero=int((dice[:, ~target] == 0).sum()))


def heldout_rows(data_dir=None, bootstrap=0):
    """Table I.1(a). A task without a bundle above the floor (MSD Heart) has no correlation row."""
    rows = []
    for task in HELDOUT_TASKS:
        panel = load_panel(('primary/' if task in PRIMARY_TASKS else 'heldout/') + task, data_dir)
        bundles = sorted(set(panel['bundle']))
        kept = [b for b in bundles if panel['dice'][panel['bundle'] == b].mean() >= 0.30]
        n, units = panel['dice'].shape[1], len(set(panel['cluster_ids']))
        row = dict(table='I.1(a)', task=task, K=len(kept), K_available=len(bundles), n_cases=n,
                   n_units=units, bootstrap=bootstrap, bootstrap_seed=0,
                   bootstrap_unit='patient' if units < n else 'volume' if task.startswith('msd_') else 'image')
        if kept:
            row.update(_correlation_row(floor_panel(panel), bootstrap))
        if 'target_empty' in panel:
            row.update(_empty_mask_counts(panel))
        rows.append(row)
    return rows


def _pixel_strata(panel):
    """analyze_pixel_overlap.strata_masks: same = same bundle or DINOv2-B x S; cross = all other pairs."""
    a, b = panel['bundle'][panel['pair_a']], panel['bundle'][panel['pair_b']]
    dino = np.isin(a, ['dinov2_vitb14', 'dinov2_vits14']) & np.isin(b, ['dinov2_vitb14', 'dinov2_vits14'])
    same_bundle = a == b
    same = same_bundle | dino
    return {'same_bundle': same_bundle, 'same_family_ckpt': dino & ~same_bundle, 'same': same,
            'cross_family': ~same, 'cross_bundle': ~same_bundle}


def _pixel_stratum_means(values, strata):
    """Per case, average the comparisons of each stratum ([case, pair] layout, as in the study)."""
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # cases without a defined comparison stay NaN
        return {s: np.nanmean(values[:, m], axis=1) for s, m in strata.items()}


def _pixel_gap_ci(row, name, a, b, idx):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        draws = np.nanmean(a[idx], axis=-1) - np.nanmean(b[idx], axis=-1)
    row[f'{name}_ci95_low'], row[f'{name}_ci95_high'] = (float(v) for v in np.nanpercentile(draws, [2.5, 97.5]))


def _pixel_row(name, panel, kind, idx):
    strata = _pixel_strata(panel)
    cond = np.ascontiguousarray(panel[f'{kind}_dice'].T, dtype=np.float64)  # NaN: both masks empty
    uncond = np.where(np.isnan(cond), 1.0, cond)                              # paper convention: empty/empty = 1
    per, per_cond = _pixel_stratum_means(uncond, strata), _pixel_stratum_means(cond, strata)
    mean = lambda v: float(np.nanmean(v))  # noqa: E731
    row = dict(table='C.3/C.4', task=name, error=kind, n_cases=cond.shape[0], K=len(set(panel['bundle'])),
               n_same_pairs=int(strata['same'].sum()), n_cross_pairs=int(strata['cross_family'].sum()),
               same=mean(per['same']), cross=mean(per['cross_family']),
               gap=mean(per['same']) - mean(per['cross_family']),
               same_bundle=mean(per['same_bundle']), same_family_ckpt=mean(per['same_family_ckpt']),
               cross_bundle=mean(per['cross_bundle']),
               gap_same_bundle_vs_cross_bundle=mean(per['same_bundle']) - mean(per['cross_bundle']),
               n_both_empty_same=int(np.isnan(cond[:, strata['same']]).sum()),
               n_comparisons_same=int(strata['same'].sum()) * cond.shape[0],
               n_both_empty_cross=int(np.isnan(cond[:, strata['cross_family']]).sum()),
               n_comparisons_cross=int(strata['cross_family'].sum()) * cond.shape[0],
               conditional_gap=mean(per_cond['same']) - mean(per_cond['cross_family']))
    if idx is not None:
        row.update(bootstrap=len(idx), bootstrap_seed=0, bootstrap_unit='image' if name != 'acdc_lv' else 'slice')
        _pixel_gap_ci(row, 'gap', per['same'], per['cross_family'], idx)
        _pixel_gap_ci(row, 'gap_same_bundle_vs_cross_bundle', per['same_bundle'], per['cross_bundle'], idx)
        _pixel_gap_ci(row, 'conditional_gap', per_cond['same'], per_cond['cross_family'], idx)
        for stratum in ('same', 'cross_family', 'same_bundle', 'same_family_ckpt'):
            column = 'cross' if stratum == 'cross_family' else stratum
            draws = np.nanmean(per[stratum][idx], axis=1)
            row[f'{column}_ci95_low'], row[f'{column}_ci95_high'] = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
        if name == 'acdc_lv' and kind == 'err':
            units = np.unique(panel['cluster_ids'])
            members = [np.flatnonzero(panel['cluster_ids'] == u) for u in units]
            picks = np.random.default_rng(0).integers(0, len(units), (len(idx), len(units)))
            gaps = [np.nanmean(per['same'][ix]) - np.nanmean(per['cross_family'][ix])
                    for ix in (np.concatenate([members[k] for k in pick]) for pick in picks)]
            row['gap_patient_ci95_low'], row['gap_patient_ci95_high'] = (float(v) for v in np.percentile(gaps, [2.5, 97.5]))
            row['n_patients'] = len(units)
    return row


def pixel_overlap_rows(data_dir=None, bootstrap=0):
    """Tables C.3-C.4. analyze_pixel_overlap.py rules: one seed-0 case-draw matrix shared by all strata;
    ACDC LV also resamples its 50 patients for the gap. kvasir_grouped is the content-group split."""
    rows = []
    for name in PIXEL_TASKS:
        panel = load_pixel_overlap(name, data_dir)
        n = panel['err_dice'].shape[1]
        idx = np.random.default_rng(0).integers(0, n, (bootstrap, n)) if bootstrap else None
        rows.extend(_pixel_row(name, panel, kind, idx) for kind in ('err', 'fp', 'fn'))
    return rows


RUNNERS = dict(zip(ANALYSES, (mask_align_rows, per_rater_rows, kvasir_grouped_rows,
                              medical_encoders_rows, amos_rows, heldout_rows, pixel_overlap_rows)))


def analysis_rows(analysis, data_dir=None, bootstrap=0):
    if analysis not in RUNNERS:
        raise ValueError(f'unknown analysis: {analysis}')
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0:
        raise ValueError('bootstrap must be a nonnegative integer')
    return RUNNERS[analysis](data_dir, bootstrap)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--analysis', choices=ANALYSES, required=True)
    parser.add_argument('--bootstrap', type=int, default=0, metavar='B',
                        help='paired draws; 0 (default) computes points only, 2000 gives the paper CIs')
    parser.add_argument('--data-dir', type=Path,
                        help='directory containing metadata.json and the NPZ packs (default: data/)')
    parser.add_argument('--output', type=Path, help='CSV destination (default: stdout)')
    args = parser.parse_args(argv)
    if args.bootstrap < 0:
        parser.error('--bootstrap must be nonnegative')
    rows = analysis_rows(args.analysis, args.data_dir, args.bootstrap)
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
