#!/usr/bin/env python3
"""End-to-end fine-tuning: per-arm within/cross statistics, matched frozen 1x1 gaps, FT_TABLE.json/.md.

Inputs under --root (one ``train_adapted.py --regime full_ft --encoder-lr ARM --out <root>/lr<ARM>`` run per arm):
  lr<ARM>/full_ft/ft_e30/{riga_cup,kvasir}/<fm>/seed_<s>.json                  riga_cup and Kvasir, content-group (grouped_v2) split
  kvasir_official/lr<ARM>/full_ft/ft_e30/kvasir/<fm>/seed_<s>.json             Kvasir official split (optional)
Frozen pairing (never cross splits): --frozen-root holds the frozen 1x1 per_case_dice/ for riga_cup and the
content-group (grouped_v2) Kvasir split; --frozen-root-official holds it for the official Kvasir split. Outputs are written under --out.

For every task-set x arm (both pre-declared arms; nothing is selected):
  1. each head: initial_state_sha256 == the primary's cache_identity.encoder_checkpoint_sha256 for that bundle (and the
     pinned expected prefix); test_ids == the frozen traces' test_ids; 11 bundles x seeds 42-45 present. Any mismatch aborts.
  2. per-arm root <out>/<set>/lr<ARM>/per_case_dice/<task>/ -> compute_all.py --only-within-cross (primary rules:
     per-bundle 0.30 floor on the mean over seeds, same-bundle + DINOv2 S/B within group, case bootstrap B=2000 seed 0).
  3. frozen matched: the frozen traces of the FT-kept bundles -> compute_all; if the frozen floor keeps a different set,
     FT and frozen are recomputed on the intersection (sensitivity).
  4. collapsed heads (head mean Dice < 0.30): Pearson degeneracy (std < 1e-8 -> NaN, excluded by the estimator), how many
     of the B=2000 case resamples make the head constant, and a sensitivity root without those heads.
Usage: compute_ft_table.py --root DIR --frozen-root DIR [--frozen-root-official DIR] [--out DIR]"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'code/src'))
from fmpool.estimators import _bootstrap_case_indices, _ci95, within_cross_rbar  # noqa: E402  compute_all's estimators
DINOV2_FAMILY_PAIRS = frozenset([frozenset({'dinov2_vitb14', 'dinov2_vits14'})])  # compute_all.DINOV2_FAMILY_PAIRS

FROZEN = {}  # 'default' / 'official' -> frozen 1x1 results root (holds per_case_dice/); set in main()
OUT = Path('.')  # output folder; set in main()
ARMS, SEEDS, FLOOR, B, EPS = ('1e-4', '1e-5'), (42, 43, 44, 45), 0.30, 2000, 1e-8
FMS = ('resnet18 resnet50 efficientnet_b0 convnext_tiny dinov2_vits14 dinov2_vitb14 mae_vitb16 deit_vitb16 clip_vitb16 '
       'clip_vitl14 biomedclip').split()
EXPECTED_PREFIX = {'resnet18': 'fbbc08d00f', 'resnet50': '1c7ce508fa', 'efficientnet_b0': '646512bdb0',  # sha256 prefixes of the pinned pretrained encoder states
                   'convnext_tiny': '685584b4b9', 'dinov2_vits14': 'd277398f92', 'dinov2_vitb14': '75ce7264ee',
                   'mae_vitb16': '26e8e213c6', 'deit_vitb16': '909cb5aae6', 'clip_vitb16': '84eeb6006c',
                   'clip_vitl14': '423a0e811f', 'biomedclip': 'bbddd91e91'}
SETS = (('riga_cup', 'riga_cup', 'lr{arm}/full_ft/ft_e30', 'default', 'riga_cup'),  # (dir, task, trace dir, frozen root key, split)
        ('kvasir_grouped_v2', 'kvasir', 'lr{arm}/full_ft/ft_e30', 'default', 'kvasir_grouped_v2'),
        ('kvasir_official', 'kvasir', 'kvasir_official/lr{arm}/full_ft/ft_e30', 'official', 'kvasir_official'))
NOTES = [
    'Both encoder-lr arms (1e-4, 1e-5) were declared before any run and both are reported; none is selected on test data.',
    'The encoder is in train() mode during fine-tuning, so BatchNorm statistics are re-estimated (ResNet18/50, EfficientNet-B0) '
    'and torchvision stochastic depth is active (ConvNeXt-T, EfficientNet-B0). This has no effect on the 7 ViT bundles.',
    'Recipe: no augmentation; constant lr with no warmup or schedule; Adam with no weight decay; no layer-wise decay; '
    'BCE loss only; head lr 1e-3; 30 epochs at batch 16; FP32 with TF32 off; final-epoch checkpoint, as in the primary.',
    'The FT scorer computes Dice in numpy float64; the primary head scorer uses on-device float32 (difference ~6e-8 relative).',
    'Content-group (grouped_v2) Kvasir FT pairs only with --frozen-root; official-split Kvasir FT pairs only with --frozen-root-official.',
    'CUDA backward in warn-only deterministic mode: FT seed replicates are not guaranteed to be bit-reproducible.',
    'ft_minus_frozen_gap: FT gap minus matched frozen gap. Both are computed on the same case-bootstrap draws (B=2000, '
    'seed 0, the draws compute_all uses for the gap CIs), each on its own floor-kept heads. The CI is the 2.5/97.5 percentile '
    'of the per-draw difference. A draw is skipped only if either gap is NaN; n_valid records how many draws were used.',
    'ft_below_frozen: bundles whose FT mean Dice (mean over seeds of per-image mean) is below the frozen mean Dice.']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def place(src: Path, dst: Path):
    """Copy (never symlink) a trace into a stats root; an existing file must be byte-identical."""
    if dst.exists():
        if dst.read_bytes() != src.read_bytes():
            raise SystemExit(f'refusing to overwrite differing trace {dst}')
        return
    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)


def stats(root: Path, task: str, traces: dict, allow_incomplete: bool = False) -> dict:
    """traces {(fm, seed): source path} -> <root>/per_case_dice copy -> compute_all within_cross JSON (+ path/sha)."""
    for (fm, seed), src in traces.items():
        place(src, root/'per_case_dice'/task/fm/f'seed_{seed}.json')
    out = root/'stats'
    cmd = [sys.executable, str(REPO/'code/scripts/compute_all.py'), '--results-root', str(root), '--out', str(out),
           '--only-within-cross', '--task', task] + (['--allow-incomplete'] if allow_incomplete else [])
    env = {**os.environ, 'PYTHONPATH': str(REPO/'code/src'), 'PYTHONDONTWRITEBYTECODE': '1'}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit(f'compute_all failed for {root}:\n{r.stderr[-2000:]}')
    f = out/'within_cross'/f'{task}.json'; d = load(f)
    if not d.get('n_within_pairs') or not d.get('n_cross_pairs'):
        raise SystemExit(f'compute_all produced no within/cross rows for {root}')
    keep = ('fms_available_before_floor', 'fms_kept_after_floor', 'within_rbar', 'cross_rbar', 'gap', 'n_within_pairs',
            'n_cross_pairs', 'n_test', 'case_bootstrap_95ci_gap', 'primary_bootstrap_95ci_gap', 'B')
    return {**{k: d.get(k) for k in keep}, 'k': len(d['fms_kept_after_floor']), 'n_heads': len(traces),
            'stats_file': str(f.relative_to(OUT)), 'stats_sha256': sha(f)}


def ft_traces(mirror: Path, task: str, frozen: Path) -> dict:
    """All 44 heads of one arm, each checked against the frozen primary (pinned state, test ids)."""
    out = {}
    for fm in FMS:
        ref = load(frozen/'per_case_dice'/task/fm/'seed_42.json')
        enc = ref['run_identity_v2']['cache_identity']['encoder_checkpoint_sha256']
        if not enc.startswith(EXPECTED_PREFIX[fm]):
            raise SystemExit(f'primary encoder sha {enc[:10]} != pinned prefix {EXPECTED_PREFIX[fm]} ({fm})')
        for s in SEEDS:
            p = mirror/task/fm/f'seed_{s}.json'; d = load(p)
            if d['run_identity_v2']['initial_state_sha256'] != enc:
                raise SystemExit(f'FT initial state != primary encoder_checkpoint_sha256: {p}')
            if d['test_ids'] != ref['test_ids'] or (d['task'], d['fm'], d['seed']) != (task, fm, s):
                raise SystemExit(f'FT trace identity/test_ids differ from the frozen primary: {p}')
            out[(fm, s)] = p
    return out


def collapsed(traces: dict, kept: list) -> list:
    rows = []
    for (fm, s), p in sorted(traces.items()):
        x = np.asarray(load(p)['per_case_dice'], dtype=np.float64)
        if x.mean() >= FLOOR:
            continue
        const = sum(float(np.std(x[idx])) < EPS for idx in _bootstrap_case_indices(len(x), B, 0))
        rows.append({'fm': fm, 'seed': s, 'mean_dice': float(x.mean()), 'std': float(x.std()), 'n_nonzero': int((x > 0).sum()),
                     'n_test': int(x.size), 'bundle_kept_after_floor': fm in kept,
                     'pearson_defined_on_full_test_set': bool(x.std() >= EPS),
                     'bootstrap_resamples_constant': int(const), 'bootstrap_B': B,
                     'estimator_treatment': 'pearson_rbar returns NaN when std < 1e-8; NaN pairs are excluded from the within/'
                                            'cross means (point estimate and each bootstrap replicate)'})
    return rows


def mean_dice(traces: dict) -> dict:
    by = {}
    for (fm, _), p in traces.items():
        by.setdefault(fm, []).append(float(np.mean(load(p)['per_case_dice'])))
    return {fm: float(np.mean(v)) for fm, v in sorted(by.items())}


def arm_row(root: Path, sdir: str, task: str, mirror_t: str, frozen: Path, split: str, arm: str) -> dict | None:
    mirror = root/mirror_t.format(arm=arm)
    if not (mirror/task).is_dir():
        return None
    base = OUT/sdir/f'lr{arm}'
    ft = ft_traces(mirror, task, frozen)
    row = {'task': task, 'split': split, 'arm': arm, 'frozen_root': str(frozen), 'ft': stats(base, task, ft)}
    kept = row['ft']['fms_kept_after_floor']
    fz = {(fm, s): frozen/'per_case_dice'/task/fm/f'seed_{s}.json' for fm in kept for s in SEEDS}
    row['frozen_matched'] = stats(base/'frozen_matched', task, fz)
    both = sorted(set(kept) & set(row['frozen_matched']['fms_kept_after_floor']))
    if both != kept:
        row['intersection'] = {'fms': both, 'ft': stats(base/'intersection_ft', task, {k: v for k, v in ft.items() if k[0] in both}),
                               'frozen': stats(base/'intersection_frozen', task, {k: v for k, v in fz.items() if k[0] in both})}
    row['collapsed_heads'] = collapsed(ft, kept)
    drop = {(c['fm'], c['seed']) for c in row['collapsed_heads'] if c['bundle_kept_after_floor']}
    if drop:
        row['excl_collapsed'] = {'excluded_heads': sorted(f'{fm}/seed_{s}' for fm, s in drop),
                                 'ft': stats(OUT/sdir/f'lr{arm}_excl_collapsed', task, {k: v for k, v in ft.items() if k not in drop},
                                               allow_incomplete=True)}  # collapsed heads dropped by design
    ftm, fzm = mean_dice(ft), mean_dice({(fm, s): frozen/'per_case_dice'/task/fm/f'seed_{s}.json' for fm in FMS for s in SEEDS})
    row['mean_dice'] = {fm: {'ft': ftm[fm], 'frozen': fzm[fm]} for fm in FMS}
    row['ft_below_frozen'] = [fm for fm in FMS if ftm[fm] < fzm[fm]]
    row['ft_below_frozen_list'] = ', '.join(row['ft_below_frozen']) or 'none'
    row['ft_minus_frozen_gap'] = paired_difference(ft, fz, kept, row['frozen_matched']['fms_kept_after_floor'])
    return row


def write_receipt(root: Path, sdir: str, arms: dict):
    """<set>/receipt.json: cells/heads collected and validated for this task-set, collapsed heads, stats files."""
    coll = root/('kvasir_official' if sdir == 'kvasir_official' else '')/'collection_receipt.json'
    if not coll.is_file():  # the collection receipt is optional
        return
    c = load(coll); task = next(iter(arms.values()))['task']
    cells = [x for x in c['cells'] if x['task'] == task]
    rec = {'task_set': sdir, 'task': task, 'collection_receipt': str(coll), 'collection_receipt_sha256': sha(coll),
           'cells_collected': sum(x['local_check']['pass'] for x in cells), 'cells_rejected': [x['cell'] for x in cells if not x['local_check']['pass']],
           'heads_collected': 4*sum(x['local_check']['pass'] for x in cells),
           'validation': ['trainer: 4 heads per cell (schema, full_ft, encoder_lr = arm, head_lr 1e-3, recipe, TF32 off, '
                          '30 ep/b16/fp32, pinned input manifest and pretrained state, test_ids = pinned split, '
                          'finite Dice, checkpoint sha)', 'collector: trace sha256 and trace checks; test_ids == '
                          'frozen primary test_ids', 'compute_ft_table: every head initial_state_sha256 == primary '
                          'encoder_checkpoint_sha256 (pinned prefixes); 11 bundles x 4 seeds per arm'],
           'collapsed_heads': {a: r['collapsed_heads'] for a, r in arms.items()},
           'stats': {a: {k: r[k]['stats_file'] for k in ('ft', 'frozen_matched')} for a, r in arms.items()}}
    (OUT/sdir/'receipt.json').write_text(json.dumps(rec, indent=1, sort_keys=True)+'\n')


def paired_difference(ft: dict, frozen: dict, ft_kept: list, fz_kept: list) -> dict:
    """FT gap - frozen gap on identical case resamples (same draws as the compute_all gap CIs)."""
    def pc(traces, kept):
        return {k: np.asarray(load(p)['per_case_dice'], dtype=np.float64) for k, p in traces.items() if k[0] in kept}
    a, z = pc(ft, ft_kept), pc(frozen, fz_kept)
    gap = lambda m: within_cross_rbar(m, family_pairs_within=DINOV2_FAMILY_PAIRS)['gap']  # noqa: E731
    diffs = []
    for idx in _bootstrap_case_indices(len(next(iter(a.values()))), B, 0):
        g = gap({k: v[idx] for k, v in a.items()}) - gap({k: v[idx] for k, v in z.items()})
        if not np.isnan(g):
            diffs.append(float(g))
    return {'point': float(gap(a) - gap(z)), 'ci95': list(_ci95(diffs)), 'n_valid': len(diffs), 'B': B, 'bootstrap_seed': 0}


def fmt(ci):
    return f'[{ci[0]:.3f}, {ci[1]:.3f}]'


def markdown(table: dict) -> str:
    L = ['# End-to-end fine-tuning + 1x1 head vs matched frozen 1x1 (generated by code/scripts/compute_ft_table.py)', '',
         '| task (split) | encoder lr | K FT | within r | cross r | FT gap [case CI] | K frozen | frozen matched gap [case CI] | '
         'FT - frozen gap [paired case CI] |', '|---|---|---|---|---|---|---|---|---|']
    for key, arms in table['rows'].items():
        for arm, r in arms.items():
            f, z = r['ft'], r['frozen_matched']
            L.append(f"| {key} | {arm} | {f['k']} | {f['within_rbar']:.3f} | {f['cross_rbar']:.3f} | {f['gap']:.3f} "
                     f"{fmt(f['primary_bootstrap_95ci_gap'])} | {z['k']} | {z['gap']:.3f} {fmt(z['primary_bootstrap_95ci_gap'])} | "
                     f"{r['ft_minus_frozen_gap']['point']:.3f} {fmt(r['ft_minus_frozen_gap']['ci95'])} |")
            if 'intersection' in r:
                i = r['intersection']
                L.append(f"| {key} (intersection {len(i['fms'])}) | {arm} | {i['ft']['k']} | | | {i['ft']['gap']:.3f} "
                         f"{fmt(i['ft']['primary_bootstrap_95ci_gap'])} | {i['frozen']['k']} | {i['frozen']['gap']:.3f} "
                         f"{fmt(i['frozen']['primary_bootstrap_95ci_gap'])} | |")
            if 'excl_collapsed' in r:
                e = r['excl_collapsed']['ft']
                L.append(f"| {key} (excl. {', '.join(r['excl_collapsed']['excluded_heads'])}) | {arm} | {e['k']} | "
                         f"{e['within_rbar']:.3f} | {e['cross_rbar']:.3f} | {e['gap']:.3f} {fmt(e['primary_bootstrap_95ci_gap'])} | | | |")
    L += ['', '## Mean Dice per bundle (mean over seeds of per-image mean Dice): FT / frozen', '']
    for key, arms in table['rows'].items():
        for arm, r in arms.items():
            L.append(f"- {key} {arm}: " + ', '.join(f"{fm} {v['ft']:.3f}/{v['frozen']:.3f}" for fm, v in r['mean_dice'].items())
                     + f" (FT below frozen: {r['ft_below_frozen_list']})")
    L += ['', '## Collapsed heads (head mean Dice < 0.30)', '']
    L += [f"- {key} {arm}: {c['fm']} seed {c['seed']} mean {c['mean_dice']:.4f}, {c['n_nonzero']}/{c['n_test']} nonzero, "
          f"std {c['std']:.4f}, Pearson defined on the test set: {c['pearson_defined_on_full_test_set']}, constant in "
          f"{c['bootstrap_resamples_constant']}/{c['bootstrap_B']} resamples (NaN pairs excluded)"
          for key, arms in table['rows'].items() for arm, r in arms.items() for c in r['collapsed_heads']] or ['- none']
    return '\n'.join(L + ['', '## Notes', ''] + [f'- {n}' for n in NOTES]) + '\n'


def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True, help='folder holding the lr<ARM>/ fine-tuning runs')
    ap.add_argument('--frozen-root', type=Path, required=True,
                    help='frozen 1x1 results root (holds per_case_dice/) for riga_cup and the content-group (grouped_v2) Kvasir split')
    ap.add_argument('--frozen-root-official', type=Path, help='frozen 1x1 results root for the official Kvasir split')
    ap.add_argument('--out', type=Path, help='output folder (default: <root>/ft_table)')
    a = ap.parse_args(argv)
    root = a.root.resolve()
    OUT = (a.out or root/'ft_table').resolve()
    FROZEN.update({k: v.resolve() for k, v in (('default', a.frozen_root), ('official', a.frozen_root_official)) if v})
    rows = {}
    for sdir, task, mirror, frozen, split in SETS:
        if frozen not in FROZEN:
            continue
        arms = {f'lr{a}': r for a in ARMS if (r := arm_row(root, sdir, task, mirror, FROZEN[frozen], split, a)) is not None}
        if arms:
            rows[sdir] = arms
    table = {'schema': 'ft_table_v1', 'script': 'code/scripts/compute_ft_table.py', 'floor': FLOOR, 'B': B,
             'bootstrap_seed': 0, 'ci_key': 'primary_bootstrap_95ci_gap (case bootstrap for riga_cup and kvasir)',
             'rows': rows, 'notes': NOTES}
    for sdir, arms in rows.items():
        write_receipt(root, sdir, arms)
    (OUT/'FT_TABLE.json').write_text(json.dumps(table, indent=1, sort_keys=True, allow_nan=False)+'\n')
    (OUT/'FT_TABLE.md').write_text(markdown(table))
    print(json.dumps({k: {a: [r['ft']['k'], round(r['ft']['gap'], 3), r['ft']['primary_bootstrap_95ci_gap'], r['frozen_matched']['k'],
                             round(r['frozen_matched']['gap'], 3), r['frozen_matched']['primary_bootstrap_95ci_gap']]
                          for a, r in v.items()} for k, v in rows.items()}))


if __name__ == '__main__':
    main()
