"""Point estimates for Table D.1 (floors, ICC, leave-out, pair-quality) and Table D.3.

Uses all four trained heads (seeds 42–45) of each retained primary bundle.
Pair means weight defined member pairs equally; cases retain their original
image/slice weighting. No resampling, confidence intervals, calibration-split
selection, or Table D.4 threshold-crossing diagnostic is performed. Undefined
estimates are written as NaN, never zero.
"""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

from correlation import _pair_masks, _point
from scores import (PRIMARY_TASKS, correlation_matrix, floor_panel, load_panel,
                    select_rows)

# Adapted from code/scripts/compute_audit_sensitivity.py,
# compute_quality_controlled_pairs.py, compute_failure_event_summary.py and
# code/src/fmpool/estimators.py.
FLOORS = (0.0, 0.20, 0.25, 0.30, 0.35, 0.40)
PERCENTILES = (25, 33, 50)


def _mean(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float('nan')


def _icc_a1(x, y):
    """Two-way absolute-agreement single-measurement ICC, two raters."""
    n = len(x)
    if n < 2:
        return float('nan')
    subject_mean = (x + y) / 2.0
    grand = float(subject_mean.mean())
    rater_mean = np.array([x.mean(), y.mean()])
    ss_subject = 2.0 * float(np.sum((subject_mean - grand) ** 2))
    ss_rater = n * float(np.sum((rater_mean - grand) ** 2))
    ss_total = float(np.sum((x - grand) ** 2) + np.sum((y - grand) ** 2))
    ss_error = ss_total - ss_subject - ss_rater
    bms = ss_subject / (n - 1)
    ems = ss_error / (n - 1)
    denominator = bms + ems + 2 * (ss_rater - ems) / n
    return float((bms - ems) / denominator) if denominator > 0 else float('nan')


def _correlation_stats(panel, statistic='pearson'):
    masks = _pair_masks(panel, 'release')
    if statistic == 'pearson':
        same, cross, gap, _, ns, nc = _point(panel['dice'], masks)
    else:
        i, j, within, between, _, _ = masks
        values = np.array([_icc_a1(panel['dice'][a], panel['dice'][b])
                           for a, b in zip(i, j)])
        same, cross = _mean(values[within]), _mean(values[between])
        gap = same - cross
        ns = int(np.sum(within & np.isfinite(values)))
        nc = int(np.sum(between & np.isfinite(values)))
    return dict(statistic=statistic, same_mean=same, cross_mean=cross, gap=gap,
                n_same_defined_pairs=ns, n_cross_defined_pairs=nc,
                n_same_pairs=int(masks[2].sum()), n_cross_pairs=int(masks[3].sum()))


def _quality_stats(panel):
    """Original six-column OLS, population case variances and finite r pairs."""
    dice = panel['dice']
    i, j, same, _, _, _ = _pair_masks(panel, 'release')
    correlations = correlation_matrix(dice)[i, j]
    valid = np.isfinite(correlations)
    means, variances = dice.mean(axis=1), dice.var(axis=1)
    x = np.column_stack((np.ones(len(i)), same,
                         (means[i] + means[j]) / 2,
                         np.abs(means[i] - means[j]),
                         (variances[i] + variances[j]) / 2,
                         np.abs(variances[i] - variances[j])))[valid]
    y = correlations[valid]
    if not len(y):
        raise ValueError('no defined pairs for pair-quality regression')
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    ss_total = float(np.sum((y - y.mean()) ** 2))
    ss_residual = float(np.sum((y - x @ beta) ** 2))
    return dict(**_correlation_stats(panel), beta_same=float(beta[1]),
                r_squared=1 - ss_residual / ss_total if ss_total > 1e-24 else float('nan'),
                n_regression_pairs=len(y),
                regression_formula='pearson_r ~ 1 + same_group + pair_mean_dice + '
                'abs_mean_dice_diff + avg_case_variance + abs_case_variance_diff',
                variance_ddof=0)


def _failure_stats(panel, percentile):
    """Pooled percentile; strict failures, defined phi pairs, all joint pairs."""
    scores = panel['dice']
    threshold = float(np.percentile(scores, percentile))
    fail = (scores < threshold).astype(np.float64)
    i, j, same, cross, _, _ = _pair_masks(panel, 'release')
    centered = fail - fail.mean(axis=1, keepdims=True)
    ss = np.sum(centered * centered, axis=1)
    denominator = np.sqrt(np.outer(ss, ss))
    with np.errstate(invalid='ignore', divide='ignore'):
        phi = (centered @ centered.T) / denominator
    phi[denominator <= 1e-12] = np.nan
    pair_phi = phi[i, j]
    joint = ((fail @ fail.T) / float(scores.shape[1]))[i, j]
    same_phi, cross_phi = _mean(pair_phi[same]), _mean(pair_phi[cross])
    same_joint, cross_joint = _mean(joint[same]), _mean(joint[cross])
    return dict(percentile=percentile, threshold=threshold,
                threshold_rule='pooled retained model-case Dice percentile; linear interpolation',
                failure_rule='score < threshold; ties are passes',
                same_fail_phi=same_phi, cross_fail_phi=cross_phi,
                gap_fail_phi=same_phi - cross_phi,
                same_joint_fail_rate=same_joint, cross_joint_fail_rate=cross_joint,
                joint_fail_rate_ratio=same_joint / cross_joint if cross_joint > 0 else float('nan'),
                n_values_equal_threshold=int(np.sum(scores == threshold)),
                n_constant_failure_members=int(np.sum(ss <= 1e-12)),
                n_same_pairs=int(same.sum()), n_cross_pairs=int(cross.sum()),
                n_same_defined_pairs=int(np.sum(same & np.isfinite(pair_phi))),
                n_cross_defined_pairs=int(np.sum(cross & np.isfinite(pair_phi))))


def _row(task, panel, analysis, floor, **values):
    return dict(task=task, analysis=analysis, panel=f'primary/{task}',
                recipe='frozen_encoder_1x1_readout', estimate='point',
                grouping='same_bundle_or_DINOv2_release_vs_other_bundle_pairs',
                aggregation='equal_member_pairs; original_image_or_slice_case_weights',
                mean_dice_floor=floor, floor_scope='bundle_mean_over_all_heads_and_cases',
                head_seeds=';'.join(map(str, sorted(set(panel['seed'])))),
                n_cases=panel['dice'].shape[1], n_units=len(set(panel['cluster_ids'])),
                n_members=len(panel['dice']), retained_K=len(set(panel['bundle'])),
                retained_bundles=';'.join(sorted(set(panel['bundle']))), **values)


def analysis_rows(analysis='all', data_dir=None):
    """Compute the advertised point estimates directly from released score panels."""
    if analysis not in ('floor', 'quality', 'threshold', 'all'):
        raise ValueError(f'unknown analysis: {analysis}')
    rows, summaries, primary_gaps = [], [], {}
    for task in PRIMARY_TASKS:
        full = load_panel(f'primary/{task}', data_dir)
        primary = floor_panel(full)
        if analysis in ('floor', 'all'):
            for floor in FLOORS:
                panel = floor_panel(full, floor)
                rows.append(_row(task, panel, 'floor', floor,
                                 **_correlation_stats(panel)))
            rows.append(_row(task, primary, 'icc', 0.30,
                             **_correlation_stats(primary, 'icc')))
            leave_out = []
            for bundle in sorted(set(primary['bundle'])):
                panel = select_rows(primary, primary['bundle'] != bundle)
                stats = _correlation_stats(panel)
                rows.append(_row(task, panel, 'leave_one_bundle_out', 0.30,
                                 omitted_bundle=bundle, **stats))
                leave_out.append(stats['gap'])
            summaries.append(_row(task, primary, 'leave_out_summary', 0.30,
                                  leave_one_bundle_out_gap_min=float(np.min(leave_out)),
                                  leave_one_bundle_out_gap_max=float(np.max(leave_out))))
            primary_gaps[task] = _correlation_stats(primary)['gap']
        if analysis in ('quality', 'all'):
            rows.append(_row(task, primary, 'quality', 0.30, **_quality_stats(primary)))
        if analysis in ('threshold', 'all'):
            for percentile in PERCENTILES:
                rows.append(_row(task, primary, 'threshold', 0.30,
                                 **_failure_stats(primary, percentile)))
    for row in summaries:
        row['leave_one_task_out_mean_gap'] = float(np.mean(
            [gap for task, gap in primary_gaps.items() if task != row['task']]))
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', choices=('floor', 'quality', 'threshold', 'all'), default='all',
                        help='floor: floor sweep, ICC(A,1) and leave-out; quality: pair-quality regression; '
                             'threshold: P25/P33/P50 failure events; all (default)')
    parser.add_argument('--data-dir', type=Path, help='directory containing the released metadata.json and NPZ packs (default: data/ in this repository)')
    parser.add_argument('--output', type=Path, help='CSV destination; default stdout')
    args = parser.parse_args(argv)
    rows = analysis_rows(args.analysis, args.data_dir)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    stream = args.output.open('w', newline='') if args.output else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            stream.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
