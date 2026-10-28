
"""Recompute common-organ case means from the upstream evaluator's scalar CSVs."""
import argparse
from pathlib import Path

import pandas as pd

import aggregate as agg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    frames = {}
    for system in agg.SYSTEMS:
        frame = pd.read_csv(args.evaluation_dir / f'{system}_per_case.csv')
        if 'system' in frame and not frame['system'].eq(system).all():
            raise ValueError(f'case-score system identity mismatch: {system}')
        # Only organ scores and explicit identity enter the finalizer, never saved macros.
        frames[system] = frame[['case', 'modality', *agg.COMMON]]
    pivot, missingness = agg.finalize_organ_frames(frames)
    args.out.mkdir(parents=True, exist_ok=False)
    pivot.to_csv(args.out / 'comparison_per_case.csv', index=False)
    agg.dump_json(args.out / 'organ_missingness.json', missingness)
    print(f'120 cases, three systems; {missingness["undefined_case_organ_cells_per_system"]} shared missing organ cells per system')


if __name__ == '__main__':
    main()
