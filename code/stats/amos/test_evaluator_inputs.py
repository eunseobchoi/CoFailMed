"""Physical-grid and label checks run before importing the external evaluator."""
import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('amos_input_aggregate', Path(__file__).with_name('aggregate.py'))
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)


def test_grid_labels_and_organ_merge(tmp_path):
    labels, predictions = tmp_path / 'labels', tmp_path / 'predictions'
    labels.mkdir(); predictions.mkdir()
    affine = np.diag([1.5, 1.5, 3., 1.])
    values = np.zeros((3, 3, 3), dtype=np.float32)
    target, prediction = labels / 'case.nii.gz', predictions / 'case_seg.nii.gz'
    nib.save(nib.Nifti1Image(values, affine), target)
    for bad_value, shifted, message in [(0., True, 'physical grid'), (.5, False, 'integer label'),
                                        (np.nan, False, 'integer label'), (41., False, 'integer label')]:
        data = values.copy(); data[0, 0, 0] = bad_value
        grid = affine.copy()
        if shifted:
            grid[0, 3] = 30.
        nib.save(nib.Nifti1Image(data, grid), prediction)
        with pytest.raises(ValueError, match=message):
            agg.run_evaluator(str(tmp_path), str(labels), str(predictions), 'mrseg_fold0')
    organs = tmp_path / 'organs'; organs.mkdir()
    for name, _ in agg.ORGANS:
        mask = values.copy()
        if name in ('spleen', 'duodenum'):
            mask[0, 0, 0] = 1
        nib.save(nib.Nifti1Image(mask, affine), organs / f'{name}.nii.gz')
    merged = tmp_path / 'merged.nii.gz'
    agg.merge_organ_masks(organs, target, merged)
    result = nib.load(merged)
    assert result.get_fdata()[0, 0, 0] == 22  # Later duodenum wins the recorded overlap order.
    np.testing.assert_array_equal(result.affine, affine)
    assert result.get_data_dtype() == np.dtype('uint8')


@pytest.mark.parametrize("override", [None, "evaluation/custom-location"])
def test_evaluator_directory_environment(monkeypatch, override):
    if override is None:
        monkeypatch.delenv("COFAILMED_AMOS_EVALUATOR", raising=False)
    else:
        monkeypatch.setenv("COFAILMED_AMOS_EVALUATOR", override)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.EVALUATOR == Path(override or "upstream/MRSegmentator/evaluation")
