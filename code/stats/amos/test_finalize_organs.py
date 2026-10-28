"""Shared missingness must be checked before averaging the common organs."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('amos_organ_aggregate', Path(__file__).with_name('aggregate.py'))
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)


def test_finite_organ_means_and_invalid_inputs():
    cases = [f'amos_{i:04d}.nii.gz' for i in range(120)]
    frames = {}
    for index, system in enumerate(agg.SYSTEMS):
        frame = pd.DataFrame(np.full((120, 13), .6 + .1 * index), columns=agg.COMMON)
        frame.loc[0, agg.COMMON[0]] = np.nan
        frame.insert(0, 'case', cases)
        frame.insert(1, 'modality', ['ct'] * 60 + ['mri'] * 60)
        frames[system] = frame
    result, missing = agg.finalize_organ_frames(frames)
    assert missing['undefined_case_organ_cells_per_system'] == 1
    assert missing['minimum_finite_organs_per_case'] == 12
    for index, system in enumerate(agg.SYSTEMS):
        np.testing.assert_allclose(result[system], .6 + .1 * index)
    # Alignment is by identity, not incidental input row order.
    shuffled = {k: v.iloc[::-1].copy() for k, v in frames.items()}
    pd.testing.assert_frame_equal(result, agg.finalize_organ_frames(shuffled)[0])
    for column, value, message in [
        (agg.COMMON[1], np.nan, 'missing masks differ'),
        (agg.COMMON[1], np.inf, 'invalid'),
        (agg.COMMON[1], 1.1, 'within'),
        ('case', cases[1], 'duplicate'),
        ('modality', 'unknown', '60 CT'),
    ]:
        bad = {k: v.copy() for k, v in frames.items()}
        bad[agg.SYSTEMS[0]].loc[0, column] = value
        with pytest.raises((ValueError, RuntimeError), match=message):
            agg.finalize_organ_frames(bad)
    empty = {k: v.copy() for k, v in frames.items()}
    for frame in empty.values():
        frame.loc[0, list(agg.COMMON)] = np.nan
    with pytest.raises(RuntimeError, match='no finite'):
        agg.finalize_organ_frames(empty)
