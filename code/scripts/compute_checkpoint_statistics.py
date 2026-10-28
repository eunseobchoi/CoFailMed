#!/usr/bin/env python3
"""Recompute the distinct-checkpoint table from scalar traces.

--primary-root and --pool-root contain per_case_dice/<task>/<bundle>/seed_*.json.
Alternatively --trace-tree points to a directory with primary/<task>/<bundle>/seed_*.json and
cross_checkpoint/<task>/<bundle>/seed_*.json (the study's JSON trace layout; this release ships the
same traces as NPZ packs instead). Outputs go only to --out. Case, patient-aggregated, and
patient-clustered slice intervals remain separate. Collection receipt metadata are not inferred from scalar traces.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import checkpoint_statistics as cs


def compute(pool_root, primary_root, tasks):
    rows = {}
    for task in tasks:
        diag = cs.ccf._task_summary(pool_root, task)
        primary = cs.ccf._task_summary(primary_root, task)
        rows[task] = cs.summarize_row(pool_root, primary_root, task, diag, primary)
        print(task, rows[task]['ckpt_gap'], rows[task]['ckpt_gap_ci95'], flush=True)
    return {'rows': rows, 'floor': cs.ccf.FUNCTIONAL_FLOOR,
            'bootstrap_b': cs.ccf.BOOTSTRAP_B, 'bootstrap_seed': cs.ccf.BOOTSTRAP_SEED,
            'producer': 'checkpoint_statistics.summarize_row',
            'note': 'Collection receipt metadata are not inferred from scalar traces.'}


def trace_tree_roots(traces, workspace, tasks):
    """Build disposable directory views; never change the released traces."""
    primary, pool = workspace / 'primary', workspace / 'pool'
    (primary / 'per_case_dice').parent.mkdir(parents=True)
    (primary / 'per_case_dice').symlink_to((traces / 'primary').resolve(), target_is_directory=True)
    for task in tasks:
        dest = pool / 'per_case_dice' / task
        dest.mkdir(parents=True)
        for family in ('primary', 'cross_checkpoint'):
            source = traces / family / task
            bundles = sorted(p for p in source.iterdir() if p.is_dir())
            expected = 11 if family == 'primary' else 6
            if len(bundles) != expected:
                raise ValueError(f'{source}: expected {expected} bundles, got {len(bundles)}')
            for bundle in bundles:
                if len(list(bundle.glob('seed_*.json'))) != 4:
                    raise ValueError(f'{bundle}: expected four decoder seed traces')
                (dest / bundle.name).symlink_to(bundle.resolve(), target_is_directory=True)
    return primary, pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-tree', type=Path,
                        help='directory with primary/ and cross_checkpoint/ JSON trace trees (study layout)')
    parser.add_argument('--primary-root', type=Path, help='results root of the 11-bundle primary pool')
    parser.add_argument('--pool-root', type=Path, help='results root of the 17-bundle expanded pool')
    parser.add_argument('--tasks', nargs='+', choices=cs.TASKS, default=list(cs.TASKS))
    parser.add_argument('--out', type=Path, required=True, help='output JSON')
    args = parser.parse_args()
    if args.trace_tree:
        if args.primary_root or args.pool_root:
            parser.error('use --trace-tree OR both explicit roots')
        with tempfile.TemporaryDirectory() as tmp:
            primary, pool = trace_tree_roots(args.trace_tree, Path(tmp), args.tasks)
            payload = compute(pool, primary, args.tasks)
    else:
        if not args.primary_root or not args.pool_root:
            parser.error('provide --trace-tree OR both --primary-root and --pool-root')
        payload = compute(args.pool_root, args.primary_root, args.tasks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
