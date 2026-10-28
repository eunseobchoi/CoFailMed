"""Load aligned Dice panels and apply the study's bundle quality screen."""
from pathlib import Path
import json
import re

import numpy as np

PRIMARY_TASKS = ('kvasir', 'acdc_lv', 'brats_wt', 'riga_cup', 'riga_disc')
PRIMARY_BUNDLES = ('biomedclip', 'clip_vitb16', 'clip_vitl14', 'convnext_tiny',
                   'deit_vitb16', 'dinov2_vitb14', 'dinov2_vits14', 'efficientnet_b0',
                   'mae_vitb16', 'resnet18', 'resnet50')
EXTRA_BUNDLES = ('clip_vitb32', 'convnext_base', 'convnext_small', 'dinov2_vitl14',
                 'resnet101', 'resnet34')
MEDICAL_BUNDLES = ('biomedparse', 'rad_dino', 'retfound_vitl16')  # Phikon is not released
PACKS = ('scores.npz', 'controls.npz', 'extensions.npz', 'pixel_overlap.npz')
AMOS_FIELDS = {'dice', 'organ_dice', 'organs', 'case_ids', 'model', 'system'}
PIXEL_FIELDS = {'err_dice', 'fp_dice', 'fn_dice', 'pair_a', 'pair_b', 'bundle', 'seed', 'case_ids', 'cluster_ids'}
CASE_FIELDS = {'case_ids', 'cluster_ids', 'case_split_ids', 'scored_export_ids', 'target_empty'}
ROW_FIELDS = {'dice', 'bundle', 'seed', 'upstream_family', 'selection_family',
              'fold', 'prediction_empty'}
EMPTY_FIELDS = {'target_empty', 'prediction_empty'}  # only MSD Spleen, which has empty targets


def validate_panel(panel):
    """Validate an explicit custom panel without imposing the released roster."""
    required = {'dice', 'bundle', 'seed', 'case_ids', 'cluster_ids',
                'upstream_family', 'selection_family'}
    missing = required - panel.keys()
    if missing:
        raise ValueError(f'missing panel fields: {sorted(missing)}')
    if panel.keys() - (CASE_FIELDS | ROW_FIELDS):
        raise ValueError('unknown panel fields')
    dice = np.asarray(panel['dice'])
    if (dice.ndim != 2 or min(dice.shape) < 1 or dice.shape[1] < 2
            or dice.dtype.kind not in 'fiu' or not np.isfinite(dice).all()
            or np.any((dice < 0) | (dice > 1))):
        raise ValueError('Dice must be a nonempty finite [member, case] matrix in [0,1]')
    for key, value in panel.items():
        value = np.asarray(value)
        if key == 'dice':
            continue
        shape = dice.shape if key == 'prediction_empty' else (
            dice.shape[1] if key in CASE_FIELDS else dice.shape[0],)
        if value.shape != shape:
            raise ValueError(f'{key}: expected shape {shape}')
        if key in EMPTY_FIELDS:
            if value.dtype != bool:
                raise ValueError(f'{key}: expected boolean flags')
        elif key in ('seed', 'fold'):
            if value.dtype.kind not in 'iu':
                raise ValueError(f'{key}: expected integer identities')
        elif value.dtype.kind != 'U' or any(not x.strip() for x in value):
            raise ValueError(f'{key}: expected nonempty string identities')
    if len(set(panel['case_ids'])) != dice.shape[1]:
        raise ValueError('duplicate case_ids')
    if len(EMPTY_FIELDS & panel.keys()) == 1:
        raise ValueError('target_empty and prediction_empty must be supplied together')
    if EMPTY_FIELDS <= panel.keys():
        target, prediction = panel['target_empty'][None, :], panel['prediction_empty']
        if np.any(dice[target & prediction] != 1) or np.any(dice[target ^ prediction] != 0):
            raise ValueError('Dice disagrees with the empty-mask convention (empty/empty = 1)')
    if 'scored_export_ids' in panel and len(set(panel['scored_export_ids'])) != dice.shape[1]:
        raise ValueError('duplicate scored_export_ids')
    members = list(zip(panel['bundle'], panel['seed'],
                       panel.get('fold', np.full(dice.shape[0], -1))))
    if len(set(members)) != len(members):
        raise ValueError('duplicate member identities')
    for bundle in set(panel['bundle']):
        mask = panel['bundle'] == bundle
        if any(len(set(panel[field][mask])) != 1
               for field in ('upstream_family', 'selection_family')):
            raise ValueError(f'{bundle}: inconsistent family labels')


def _family(bundle):
    for prefix in ('dinov2', 'clip', 'resnet', 'convnext', 'efficientnet', 'mae', 'deit'):
        if bundle.startswith(prefix):
            return prefix
    return bundle


def _read_entry(name, data_dir):
    """Return the metadata entry and raw NPZ fields of one panel (no pickle)."""
    root = Path(data_dir) if data_dir is not None else Path(__file__).parent / 'data'
    metadata = json.loads((root / 'metadata.json').read_text())
    matches = [entry for entry in metadata['panels'] if entry['key'] == name]
    if len(matches) != 1:
        raise ValueError(f'unknown or duplicate panel: {name}')
    entry = matches[0]
    if entry['pack'] not in PACKS:
        raise ValueError(f'{name}: unsupported score pack')
    with np.load(root / entry['pack'], allow_pickle=False) as pack:
        if len(pack.files) != len(set(pack.files)):
            raise ValueError('duplicate NPZ keys')
        panel = {key[len(name) + 1:]: pack[key]
                 for key in pack.files if key.startswith(name + '/')}
    return entry, panel


def load_panel(name, data_dir=None):
    """Read a complete released panel from metadata.json and its NPZ (no pickle)."""
    if name.startswith('amos/'):
        raise ValueError('AMOS panels have their own schema; use load_amos')
    entry, panel = _read_entry(name, data_dir)
    task = {'nnunet/acdc_3d': 'acdc_lv', 'nnunet/riga_cup_2d': 'riga_cup'}.get(
        name, 'kvasir' if name.startswith('kvasir_grouped/') else name.rsplit('/', 1)[-1])
    if entry['task'] != task:
        raise ValueError(f'{name}: task identity disagrees with metadata')
    validate_panel(panel)
    rows, cases = panel['dice'].shape
    if ((rows, cases) != (entry['n_traces'], entry['n_cases'])
            or len(set(panel['bundle'])) != entry['n_bundles']
            or len(set(panel['cluster_ids'])) != entry['n_resampling_units']):
        raise ValueError(f'{name}: dimensions disagree with metadata')
    bundles, seeds = PRIMARY_BUNDLES, tuple(range(42, 46))
    if name.startswith('cross_checkpoint/') or name == 'kvasir_grouped/cross_checkpoint':
        bundles = EXTRA_BUNDLES
    elif name.startswith('medical/'):
        bundles = MEDICAL_BUNDLES
    elif name.startswith('random_init/'):
        bundles, seeds = ('convnext_tiny', 'resnet50'), tuple(range(42, 50))
    elif name.startswith('nnunet/'):
        bundles = ('nnunet_3d_fullres_100ep' if entry['task'] == 'acdc_lv'
                   else 'nnunet_2d_100ep',)
        seeds = (13, 37)
    folds = tuple(range(5)) if name.startswith('nnunet/') else (-1,)
    actual = set(zip(panel['bundle'], panel['seed'], panel.get('fold', np.full(rows, -1))))
    expected = {(b, s, f) for b in bundles for s in seeds for f in folds}
    if actual != expected:
        raise ValueError(f'{name}: incomplete or unexpected roster; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    if set(entry['seeds']) != set(seeds) or set(entry['folds']) != (set(folds) - {-1}):
        raise ValueError(f'{name}: metadata seed/fold contract mismatch')
    for bundle, upstream, selection in zip(panel['bundle'], panel['upstream_family'], panel['selection_family']):
        family = _family(bundle)
        if selection != family or upstream != ('openai_clip' if family == 'clip' else family):
            raise ValueError(f'{name}: incorrect family for {bundle}')
    ids = panel['case_ids']
    if entry['task'] == 'acdc_lv':
        units = [case.split('_frame', 1)[0] for case in ids]
    elif entry['task'] == 'brats_wt':
        matches = [re.fullmatch(r'(BraTS-GLI-\d{5})-\d{3}_slice\d+', case) for case in ids]
        if not all(matches):
            raise ValueError(f'{name}: invalid BraTS patient IDs')
        units = [match[1] for match in matches]
    else:
        units = ids
    if not np.array_equal(panel['cluster_ids'], units):
        raise ValueError(f'{name}: cluster_ids disagree with case identities')
    return panel


def load_amos(modality, data_dir=None):
    """Read one AMOS22 panel: three models x 60 volumes, with 13 organ scores."""
    if modality not in ('ct', 'mri'):
        raise ValueError('modality must be ct or mri')
    entry, panel = _read_entry(f'amos/{modality}', data_dir)
    if panel.keys() != AMOS_FIELDS or entry['task'] != f'amos22_{modality}':
        raise ValueError(f'amos/{modality}: unexpected fields or task')
    dice, organs = panel['dice'], panel['organ_dice']
    shape = (entry['n_traces'], entry['n_cases'])
    if (dice.shape != shape or organs.shape != (*shape, entry['n_organs'])
            or dice.dtype != np.float64 or organs.dtype != np.float64
            or panel['organs'].shape != (entry['n_organs'],)
            or panel['case_ids'].shape != (shape[1],) or len(set(panel['case_ids'])) != shape[1]
            or panel['model'].shape != (shape[0],) or panel['system'].shape != (shape[0],)):
        raise ValueError(f'amos/{modality}: dimensions disagree with metadata')
    finite = np.isfinite(organs)
    if (not np.isfinite(dice).all() or np.any((dice < 0) | (dice > 1))
            or np.any((organs[finite] < 0) | (organs[finite] > 1))
            or not all(np.array_equal(finite[0], f) for f in finite) or not finite.any(axis=2).all()):
        raise ValueError(f'amos/{modality}: invalid scores or unequal missingness')
    if np.abs(np.nanmean(organs, axis=2) - dice).max() > 1e-12:
        raise ValueError(f'amos/{modality}: case score is not the mean of finite organ scores')
    if list(panel['system']) != ['mrsegmentator', 'mrsegmentator', 'totalsegmentator']:
        raise ValueError(f'amos/{modality}: expected two MRSegmentator folds, then TotalSegmentator')
    return panel


def load_pixel_overlap(name, data_dir=None):
    """Read one error-mask overlap panel: [pair, case] Dice of two heads' error masks (NaN = both empty)."""
    entry, panel = _read_entry(f'pixel_overlap/{name}', data_dir)
    if panel.keys() != PIXEL_FIELDS:
        raise ValueError(f'pixel_overlap/{name}: unexpected fields')
    shape = (entry['n_pairs'], entry['n_cases'])
    a, b, m = panel['pair_a'], panel['pair_b'], entry['n_heads']
    if (any(panel[k].shape != shape or panel[k].dtype.kind != 'f' for k in ('err_dice', 'fp_dice', 'fn_dice'))
            or a.shape != (shape[0],) or b.shape != (shape[0],) or panel['bundle'].shape != (m,)
            or panel['seed'].shape != (m,) or panel['case_ids'].shape != (shape[1],)
            or panel['cluster_ids'].shape != (shape[1],) or len(set(panel['case_ids'])) != shape[1]):
        raise ValueError(f'pixel_overlap/{name}: dimensions disagree with metadata')
    if (not np.array_equal(np.stack([a, b]), np.stack(np.triu_indices(m, 1)))
            or len(set(zip(panel['bundle'], panel['seed']))) != m):
        raise ValueError(f'pixel_overlap/{name}: pairs must list every head pair once, in upper-triangle order')
    for key in ('err_dice', 'fp_dice', 'fn_dice'):
        values = panel[key][np.isfinite(panel[key])]
        if np.any((values < 0) | (values > 1)):
            raise ValueError(f'pixel_overlap/{name}: {key} outside [0, 1]')
    empty = np.isnan(panel['err_dice'])
    if not (np.isnan(panel['fp_dice'][empty]).all() and np.isnan(panel['fn_dice'][empty]).all()):
        raise ValueError(f'pixel_overlap/{name}: empty error masks must have empty FP and FN masks')
    return panel


def select_rows(panel, mask):
    """Select members while preserving case metadata, including optional fields."""
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != (len(panel['dice']),):
        raise ValueError('row mask must be a boolean vector aligned with members')
    return {key: value.copy() if key in CASE_FIELDS else value[mask].copy()
            for key, value in panel.items()}


def floor_panel(panel, floor=0.30):
    """Keep every head of bundles whose mean over heads and cases meets floor."""
    validate_panel(panel)
    if not np.isfinite(floor) or not 0 <= floor <= 1:
        raise ValueError('floor must be finite and in [0,1]')
    keep = {bundle for bundle in set(panel['bundle'])
            if panel['dice'][panel['bundle'] == bundle].mean() >= floor}
    return select_rows(panel, np.isin(panel['bundle'], list(keep)))


def expanded_panel(task, data_dir=None, *, grouped=False):
    """Combine primary and added checkpoints on exactly the same case columns.

    grouped=True uses the content-group Kvasir split (task must be kvasir).
    """
    if task not in PRIMARY_TASKS:
        raise ValueError(f'unknown primary task: {task}')
    if grouped and task != 'kvasir':
        raise ValueError('the content-group split exists only for kvasir')
    names = (('kvasir_grouped/primary', 'kvasir_grouped/cross_checkpoint') if grouped
             else (f'primary/{task}', f'cross_checkpoint/{task}'))
    primary, extra = (load_panel(name, data_dir) for name in names)
    if primary.keys() != extra.keys():
        raise ValueError('expanded panels have different fields')
    for key in primary.keys() & CASE_FIELDS:
        if not np.array_equal(primary[key], extra[key]):
            raise ValueError(f'expanded panels disagree on {key}')
    result = {key: value.copy() if key in CASE_FIELDS else np.concatenate([value, extra[key]])
              for key, value in primary.items()}
    validate_panel(result)
    return result


def correlation_matrix(dice):
    """Pearson matrix; pairs involving std < 1e-8 are undefined (NaN)."""
    dice = np.asarray(dice, dtype=float)
    if dice.ndim != 2 or dice.shape[1] < 2 or not np.isfinite(dice).all():
        raise ValueError('correlation requires finite [member, case] values with >=2 cases')
    matrix = np.full((len(dice), len(dice)), np.nan)
    valid = np.flatnonzero(dice.std(axis=1) >= 1e-8)
    if len(valid):
        matrix[np.ix_(valid, valid)] = np.atleast_2d(np.corrcoef(dice[valid]))
    return matrix
