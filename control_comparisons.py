"""Table G.3 (four released rows) and Table G.5 from aligned public scores.

Paired FT-minus-frozen gaps use recipe-specific 0.30 floors and shared seed-0
case draws. The two grouped-Kvasir sensitivity rows are not in the score pack.
nnU-Net effective size uses five folds at training seed 13, their pooled P33,
and fixed-threshold resampling (ACDC patients, RIGA images). --bootstrap 2000
reproduces paper CIs; the default 0 computes points and observed support only.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

# Adapted from code/scripts/compute_ft_table.py (paired_difference) and
# code/scripts/compute_nnunet_case_identical.py (_effective_size_summary).
from correlation import _pair_masks, _point, correlate
from effective_size import _interval, _pool_summary
from scores import CASE_FIELDS, floor_panel, load_panel, select_rows, validate_panel


def paired_gap(ft, frozen, bootstrap=0):
    """Compare complete aligned panels, fixing each recipe's selected pool."""
    _validate_bootstrap(bootstrap)
    for panel in (ft, frozen):
        validate_panel(panel)
        if 'fold' in panel:
            raise ValueError('Table G.3 compares encoder/readout panels, not fold-bearing panels')
    for key in (ft.keys() | frozen.keys()) & CASE_FIELDS:
        if key not in ft or key not in frozen or not np.array_equal(ft[key], frozen[key]):
            raise ValueError(f'FT/frozen {key} differ; case pairing is required')
    if set(zip(ft['bundle'], ft['seed'])) != set(zip(frozen['bundle'], frozen['seed'])):
        raise ValueError('FT/frozen source members differ')
    if len(set(ft['cluster_ids'])) != len(ft['case_ids']):
        raise ValueError('Table G.3 supports image panels only')
    ft, frozen = floor_panel(ft), floor_panel(frozen)
    a, z = correlate(ft), correlate(frozen)
    result = dict(n_cases=a['n_cases'], K_ft=a['retained_K'], K_frozen=z['retained_K'],
                  kept_ft=';'.join(sorted(set(ft['bundle']))),
                  kept_frozen=';'.join(sorted(set(frozen['bundle']))),
                  ft_gap=a['gap'], frozen_gap=z['gap'],
                  gap_difference=a['gap'] - z['gap'], bootstrap=bootstrap,
                  bootstrap_seed=0, bootstrap_unit='image')
    # Same draw for both recipes, as compute_ft_table.paired_difference.
    masks_a, masks_z = _pair_masks(ft, 'release'), _pair_masks(frozen, 'release')
    rng = np.random.default_rng(0)
    draws = np.full((bootstrap, 2), np.nan)
    for i in range(bootstrap):
        idx = rng.integers(0, a['n_cases'], a['n_cases'])
        draws[i] = (_point(ft['dice'][:, idx], masks_a)[2],
                    _point(frozen['dice'][:, idx], masks_z)[2])
    if bootstrap:
        result.update(_interval(draws[:, 0], 'ft_gap'))
        result.update(_interval(draws[:, 1], 'frozen_gap'))
        result.update(_interval(draws[:, 0] - draws[:, 1], 'gap_difference'))
    return result


def nnunet_size(panel, bootstrap=0, *, seed, clustered):
    """One seed-13 five-fold pool; threshold fixed before resampling."""
    _validate_bootstrap(bootstrap)
    validate_panel(panel)
    if 'fold' not in panel:
        raise ValueError('nnU-Net requires fold identities')
    if len(set(panel['bundle'])) != 1:
        raise ValueError('nnU-Net folds must belong to one model bundle')
    selected = select_rows(panel, panel['seed'] == 13)
    if len(selected['dice']) != 5 or set(selected['fold']) != set(range(5)):
        raise ValueError('nnU-Net requires exactly one seed-13 member per fold 0..4')
    dice = selected['dice']
    threshold = float(np.percentile(dice, 33))
    pools = np.arange(5)[None, :]
    summary = _pool_summary(dice, threshold, pools)
    failures = (dice < threshold).sum(axis=1)
    result = dict(training_seed=13, nominal_M=5, pooled_P33=threshold,
                  **summary, member_folds=json.dumps(selected['fold'].tolist()),
                  member_failures=json.dumps(failures.tolist()),
                  member_passes=json.dumps((dice.shape[1] - failures).tolist()),
                  bootstrap=bootstrap, bootstrap_seed=seed,
                  bootstrap_unit='patient' if clustered else 'image')
    groups = ([np.flatnonzero(selected['cluster_ids'] == label)
               for label in sorted(set(selected['cluster_ids']))] if clustered else None)
    if groups is not None and len(groups) < 2:
        raise ValueError('patient bootstrap requires at least two patients')
    rng = np.random.default_rng(seed)
    draws = np.full(bootstrap, np.nan)
    for i in range(bootstrap):
        idx = (rng.integers(0, dice.shape[1], dice.shape[1]) if groups is None else
               np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]))
        draws[i] = _pool_summary(dice[:, idx], threshold, pools)['M_eff']
    if bootstrap:
        result.update(_interval(draws, 'M_eff'))
    return result


def _validate_bootstrap(bootstrap):
    if (isinstance(bootstrap, (bool, np.bool_))
            or not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0):
        raise ValueError('bootstrap must be a nonnegative integer')


def analysis_rows(analysis='all', data_dir=None, bootstrap=0):
    """Recompute only Table G.3/G.5 rows represented by the released panels."""
    _validate_bootstrap(bootstrap)
    if analysis not in ('paired', 'nnunet', 'all'):
        raise ValueError(f'unknown analysis: {analysis}')
    rows = []
    if analysis in ('paired', 'all'):
        for task in ('kvasir', 'riga_cup'):
            frozen = load_panel(f'primary/{task}', data_dir)
            for rate in ('1e-4', '1e-5'):
                ft = load_panel(f'ft/lr{rate}/{task}', data_dir)
                rows.append(dict(table='G.3', task=task, recipe=f'ft_lr{rate}',
                                 **paired_gap(ft, frozen, bootstrap)))
    if analysis in ('nnunet', 'all'):
        for task, name, seed in (('acdc_lv', 'acdc_3d', 1_120_000),
                                 ('riga_cup', 'riga_cup_2d', 1_020_000)):
            panel = load_panel(f'nnunet/{name}', data_dir)
            correlation = correlate(panel, 'fold', bootstrap=bootstrap)
            rows.append(dict(table='G.5', task=task, recipe='nnunet_100ep',
                             n_cases=correlation['n_cases'], n_units=correlation['n_units'],
                             mean_dice=correlation['mean_dice'],
                             same_fold=correlation['within'], cross_fold=correlation['cross'],
                             fold_gap=correlation['gap'],
                             fold_gap_ci95_low=correlation['gap_ci_low'],
                             fold_gap_ci95_high=correlation['gap_ci_high'],
                             fold_gap_n_valid=correlation['n_valid_boot'],
                             fold_gap_bootstrap_seed=0,
                             **nnunet_size(panel, bootstrap, seed=seed,
                                           clustered=task == 'acdc_lv')))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', choices=('paired', 'nnunet', 'all'), default='all',
                        help='paired: Table G.3 FT-minus-frozen gaps; nnunet: Table G.5; all (default)')
    parser.add_argument('--bootstrap', type=int, default=0, metavar='B',
                        help='paired draws; 0 (default) computes points only, 2000 gives the percentile CIs')
    parser.add_argument('--data-dir', type=Path, help='directory containing the released metadata.json and NPZ packs (default: data/ in this repository)')
    parser.add_argument('--output', type=Path, help='CSV destination (default stdout)')
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
