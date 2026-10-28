
"""Run the pinned upstream AMOS evaluator on separately acquired volumetric masks."""
import argparse
from pathlib import Path

import aggregate as agg
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--segmentation', type=Path, required=True)
    parser.add_argument('--system', choices=agg.SYSTEMS, required=True)
    parser.add_argument('--evaluator', type=Path, required=True)
    parser.add_argument('--case-table', type=Path, required=True,
                        help='CSV containing exactly the evaluated case IDs and modalities')
    parser.add_argument('--totalseg-organs', action='store_true',
                        help='segmentation contains one binary-organ directory per case')
    parser.add_argument('--images', type=Path, help='original image grid for merging TotalSegmentator organs')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.totalseg_organs:
        if args.system != 'totalseg' or args.images is None:
            parser.error('--totalseg-organs requires --system totalseg and --images')
        directories = sorted(p for p in args.segmentation.iterdir() if p.is_dir())
        if not directories:
            raise ValueError('no organ-mask case directories')
        merged = args.out / 'totalseg_merged'
        merged.mkdir(parents=True, exist_ok=False)
        for directory in directories:
            agg.merge_organ_masks(directory, args.images / f'{directory.name}.nii.gz',
                                  merged / f'{directory.name}_seg.nii.gz')
        args.segmentation = merged
    names = sorted(p.name for p in args.segmentation.glob('*.nii.gz'))
    if not names or any(not (args.labels / n.replace('_seg', '')).is_file() for n in names):
        raise ValueError('each prediction must have a matching ground-truth NIfTI')
    if len({n.replace('_seg', '') for n in names}) != len(names):
        raise ValueError('duplicate normalized prediction case IDs')
    identity = pd.read_csv(args.case_table)[['case', 'modality']]
    if (identity['case'].isna().any() or identity['case'].duplicated().any()
            or not identity['modality'].isin(['ct', 'mri']).all()
            or set(identity['case']) != {n.replace('_seg', '') for n in names}):
        raise ValueError('prediction cohort differs from unique case/modality table')
    identity = identity.set_index('case').loc[[n.replace('_seg', '') for n in names]].reset_index()
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / f'{args.system}_{suffix}.csv').exists() for suffix in ('dsc', 'hd', 'per_case')):
        raise FileExistsError('refusing to overwrite evaluator results')
    agg.EVALUATOR = args.evaluator
    agg.run_evaluator(str(args.out.resolve()), str(args.labels.resolve()),
                      str(args.segmentation.resolve()), args.system)
    scores = pd.read_csv(args.out / f'{args.system}_dsc.csv')
    if len(scores) != len(identity):
        raise ValueError('evaluator row count differs from prediction cohort')
    identity['system'] = args.system
    pd.concat([identity, scores], axis=1).to_csv(args.out / f'{args.system}_per_case.csv', index=False)


if __name__ == '__main__':
    main()
