"""Table 2 points, Fig. 3(a,b) tiers, Table 3 ViTs, and Fig. 5 control levels.

Optional paired percentile intervals condition on the observed floor-selected pool.
--hierarchical adds Table 2's occurrence-local panel-resampling sensitivity.
Figure rendering is not included.
"""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

from scores import (PRIMARY_TASKS, correlation_matrix, expanded_panel, floor_panel,
                    load_panel, select_rows, validate_panel)

VITS = ('dinov2_vitb14', 'biomedclip', 'mae_vitb16', 'deit_vitb16', 'clip_vitb16')


def _pair_masks(panel, grouping):
    i, j = np.triu_indices(len(panel['dice']), 1)
    same = panel['bundle'][i] == panel['bundle'][j]
    if grouping == 'release':
        dino = np.isin(panel['bundle'], ['dinov2_vitb14', 'dinov2_vits14'])
        within, cross = same | (dino[i] & dino[j]), ~(same | (dino[i] & dino[j]))
        labels = ('same_bundle_or_DINOv2_release', 'other_bundle_pairs')
    elif grouping == 'checkpoint':
        within, cross = same, ~same
        labels = ('same_bundle', 'distinct_bundles')
    elif grouping == 'family':
        family = panel['upstream_family'][i] == panel['upstream_family'][j]
        within, cross = family & ~same, ~family
        labels = ('same_family_distinct_checkpoint', 'cross_family')
    elif grouping == 'fold' and 'fold' in panel:
        within = panel['fold'][i] == panel['fold'][j]
        cross = ~within
        labels = ('same_fold', 'different_fold')
    else:
        raise ValueError(f'unsupported grouping or missing fold: {grouping}')
    return i, j, within, cross, same, labels


def _point(dice, masks):
    i, j, within, cross, same, _ = masks
    values = correlation_matrix(dice)[i, j]
    def mean(mask):
        finite = values[mask & np.isfinite(values)]
        return float(finite.mean()) if len(finite) else float('nan')
    a, b = mean(within), mean(cross)
    return a, b, a-b, mean(same), int(np.sum(within & np.isfinite(values))), int(np.sum(cross & np.isfinite(values)))


def correlate(panel, grouping='release', *, bootstrap=0, seed=0, unit='auto'):
    """Score an explicit panel; apply floor_panel separately when required.

    family compares distinct checkpoints in one family with cross-family pairs;
    shared-checkpoint levels are returned separately. Custom rosters are supported.
    """
    validate_panel(panel)
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0:
        raise ValueError('bootstrap must be a nonnegative integer')
    if unit not in ('auto', 'case', 'patient'):
        raise ValueError('unit must be auto, case, or patient')
    masks = _pair_masks(panel, grouping)
    a, b, gap, shared, na, nb = _point(panel['dice'], masks)
    if not np.isfinite(gap):
        raise ValueError('correlation contrast has an empty or undefined pair stratum')
    n = panel['dice'].shape[1]
    clustered = len(set(panel['cluster_ids'])) < n
    unit = ('patient' if clustered else 'case') if unit == 'auto' else unit
    if unit == 'patient' and not clustered:
        raise ValueError('panel does not provide repeated patient clusters; use case resampling')
    groups = ([np.flatnonzero(panel['cluster_ids'] == label)
               for label in sorted(set(panel['cluster_ids']))] if unit == 'patient' else None)
    if groups is not None and len(groups) < 2:
        raise ValueError('patient bootstrap requires at least two patients')
    result = dict(n_cases=n, retained_K=len(set(panel['bundle'])), n_members=len(panel['dice']),
                  n_units=len(groups) if groups is not None else n, grouping=grouping,
                  within_group=masks[-1][0], cross_group=masks[-1][1],
                  within=a, cross=b, gap=gap, mean_dice=float(panel['dice'].mean()),
                  same_checkpoint=shared if grouping == 'family' else '',
                  n_within_pairs=na, n_cross_pairs=nb,
                  n_within_pairs_total=int(masks[2].sum()), n_cross_pairs_total=int(masks[3].sum()),
                  bootstrap=bootstrap, bootstrap_seed=seed, ci_unit=unit if bootstrap else '',
                  n_valid_boot=0, gap_ci_low='', gap_ci_high='')
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        idx = (rng.integers(0, n, n) if groups is None else
               np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]))
        draw = _point(panel['dice'][:, idx], masks)[2]
        if np.isfinite(draw):
            draws.append(draw)
    if bootstrap:
        result['n_valid_boot'] = len(draws)
        if draws:
            result['gap_ci_low'], result['gap_ci_high'] = np.percentile(draws, [2.5, 97.5])
    return result


def hierarchical_sensitivity(panel, *, bootstrap=2000, seed=0, unit='auto', drop_self=False):
    """Table 2 descriptive percentiles; not a family-population interval.

    Collapse DINOv2-B/S into one release group. Resample groups, then heads
    within each group occurrence. Repeated heads retain self-correlations
    (drop_self=True excludes them: the Appendix C.1 sensitivity);
    pairs between separate occurrences of the same group are excluded.
    The observed floor is fixed and case/patient draws are shared by all heads.
    """
    validate_panel(panel)
    if (isinstance(bootstrap, (bool, np.bool_))
            or not isinstance(bootstrap, (int, np.integer)) or bootstrap < 1):
        raise ValueError('bootstrap must be a positive integer')
    if unit not in ('auto', 'case', 'patient'):
        raise ValueError('unit must be auto, case, or patient')
    if 'dinov2' in panel['bundle']:
        raise ValueError("bundle label 'dinov2' is reserved for the merged DINOv2-B/S release")
    labels = np.asarray(['dinov2' if b in ('dinov2_vitb14', 'dinov2_vits14')
                         else b for b in panel['bundle']])
    seeds = panel['seed'] + 10_000 * (panel['bundle'] == 'dinov2_vits14')
    families = np.unique(labels)
    if len(families) < 2:
        raise ValueError('hierarchical sensitivity requires at least two release groups')
    members = {f: np.flatnonzero(labels == f)[np.argsort(seeds[labels == f])]
               for f in families}
    n = panel['dice'].shape[1]
    clustered = len(np.unique(panel['cluster_ids'])) < n
    unit = ('patient' if clustered else 'case') if unit == 'auto' else unit
    if unit == 'patient' and not clustered:
        raise ValueError('panel does not provide repeated patient clusters')
    groups = ([np.flatnonzero(panel['cluster_ids'] == label)
               for label in np.unique(panel['cluster_ids'])] if unit == 'patient' else None)
    if groups is not None and len(groups) < 2:
        raise ValueError('patient bootstrap requires at least two patients')
    case_seed, panel_seed = np.random.SeedSequence(seed).spawn(2)
    case_rng, panel_rng = np.random.default_rng(case_seed), np.random.default_rng(panel_seed)
    gaps = []
    for _ in range(bootstrap):
        cases = (case_rng.integers(0, n, n) if groups is None else
                 np.concatenate([groups[k] for k in
                                 case_rng.integers(0, len(groups), len(groups))]))
        sampled = panel_rng.choice(families, size=len(families), replace=True)
        rows, occurrences, drawn_labels = [], [], []
        for occurrence, family in enumerate(sampled):
            chosen = panel_rng.choice(members[family], size=len(members[family]), replace=True)
            rows.extend(chosen)
            occurrences.extend([occurrence] * len(chosen))
            drawn_labels.extend([family] * len(chosen))
        i, j = np.triu_indices(len(rows), 1)
        occurrences, drawn_labels = np.asarray(occurrences), np.asarray(drawn_labels)
        values = correlation_matrix(panel['dice'][rows][:, cases])[i, j]
        finite = np.isfinite(values)
        same_occurrence = occurrences[i] == occurrences[j]
        if drop_self:
            same_occurrence &= np.asarray(rows)[i] != np.asarray(rows)[j]
        within = values[finite & same_occurrence]
        cross = values[finite & (drawn_labels[i] != drawn_labels[j])]
        if len(within) and len(cross):
            gaps.append(float(within.mean() - cross.mean()))
    low, high = np.percentile(gaps, [2.5, 97.5]) if gaps else (np.nan, np.nan)
    return dict(hierarchical_bootstrap=bootstrap, hierarchical_seed=seed,
                hierarchical_unit=unit, hierarchical_n_valid=len(gaps),
                hierarchical_gap_low=float(low), hierarchical_gap_high=float(high))


def analysis_rows(analysis='all', data_dir=None, bootstrap=0, unit='auto'):
    """Return named paper contrasts from the released panels."""
    if analysis not in ('all', 'primary', 'checkpoints', 'vit', 'controls'):
        raise ValueError(f'unknown analysis: {analysis}')
    rows = []
    def add(name, panel, grouping='release', ci_unit=unit, reference=''):
        rows.append(dict(panel=name, cross_bundle_reference=reference,
                         **correlate(panel, grouping, bootstrap=bootstrap, unit=ci_unit)))
    if analysis in ('all', 'primary', 'vit', 'controls'):
        for task in PRIMARY_TASKS:
            primary = floor_panel(load_panel(f'primary/{task}', data_dir))
            if analysis in ('all', 'primary', 'controls'):
                add(f'primary/{task}', primary)
            if analysis in ('all', 'vit'):
                vit = select_rows(primary, np.isin(primary['bundle'], VITS))
                reference = correlate(primary)['cross']
                add(f'vit/{task}', vit, 'checkpoint', 'case' if unit == 'auto' else unit, reference)
    if analysis in ('all', 'checkpoints'):
        for task in PRIMARY_TASKS:
            add(f'expanded17/{task}', floor_panel(expanded_panel(task, data_dir)), 'family')
    if analysis in ('all', 'controls'):
        names = ([f'unet_skip/{task}' for task in PRIMARY_TASKS if task != 'brats_wt']
                 + [f'ft/lr{lr}/{task}' for lr in ('1e-4', '1e-5') for task in ('kvasir', 'riga_cup')]
                 + [f'random_init/{task}' for task in ('acdc_lv', 'riga_cup')]
                 + ['nnunet/acdc_3d', 'nnunet/riga_cup_2d'])
        for name in names:
            panel = load_panel(name, data_dir)
            grouping = 'fold' if name.startswith('nnunet/') else 'checkpoint' if name.startswith('random_init/') else 'release'
            add(name, panel if grouping == 'fold' else floor_panel(panel), grouping)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', choices=('primary', 'checkpoints', 'vit', 'controls', 'all'), default='primary',
                        help='primary: Table 2 gaps (default); checkpoints: 17-bundle tiers; vit: five-ViT '
                             'comparison; controls: primary rows plus the Figure 5 controls; all: every panel')
    parser.add_argument('--data-dir', type=Path, help='directory containing the released metadata.json and NPZ packs (default: data/ in this repository)')
    parser.add_argument('--bootstrap', type=int, default=0, help='paired draws; 0 (default) computes points only, 2000 gives the percentile CIs')
    parser.add_argument('--unit', choices=('auto', 'case', 'patient'), default='auto',
                        help='resampling unit; auto (default): patient clusters where present, cases for the five-ViT rows')
    parser.add_argument('--output', type=Path, help='CSV destination; default stdout')
    parser.add_argument('--hierarchical', action='store_true',
                        help='add Table 2 panel-resampling sensitivity (primary analysis only)')
    args = parser.parse_args(argv)
    if args.bootstrap < 0:
        parser.error('--bootstrap must be nonnegative')
    if args.hierarchical and (args.analysis != 'primary' or args.bootstrap == 0):
        parser.error('--hierarchical requires --analysis primary and --bootstrap > 0')
    rows = analysis_rows(args.analysis, args.data_dir, args.bootstrap, args.unit)
    if args.hierarchical:
        for row in rows:
            panel = floor_panel(load_panel(row['panel'], args.data_dir))
            row.update(hierarchical_sensitivity(panel, bootstrap=args.bootstrap, unit=args.unit))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    stream = args.output.open('w', newline='') if args.output else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            stream.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
