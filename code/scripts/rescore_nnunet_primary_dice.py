#!/usr/bin/env python3
"""Re-score collected case-identical nnU-Net holdout predictions with the primary Dice convention.

The nnU-Net scorer (train_nnunet_baseline._binary_dice) uses 2|P&G| / (|P|+|G|+1e-6) with Dice(empty, empty)=1;
the primary rows use fmpool.estimators.dice_per_case (no epsilon, same empty/empty=1). For every trace
``<task>_fold{F}_seed{S}.json`` this reads ``<cells>/<trace>.cell/holdout_predictions`` and the holdout labels,
writes a same-schema trace with primary-convention Dice to ``--out-dir`` (so compute_nnunet_case_identical.py
can summarise it) and a ``rescore_report.json`` with empty-case counts and the largest per-case difference.
NIfTI geometry is validated using the recorded nnU-Net SimpleITKIO reader's
size, spacing, origin and direction. No geometry mismatch bypass is supported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fmpool import determinism, provenance
from fmpool.estimators import dice_per_case
from predict_nnunet_holdout import _release_case_id
from train_nnunet_baseline import (
    _label_files, _same_ids, _tree_hashes, _unique_ids,
    _validate_masks, _validate_nifti_pair, _write_json,
)


def _read_pair(pred_path: Path, gt_path: Path):
    if pred_path.name != gt_path.name:
        raise ValueError(f"prediction/GT file format mismatch: {pred_path.name}")
    if pred_path.name.endswith(".png"):
        from PIL import Image
        with Image.open(pred_path) as image:
            pred = np.array(image)
        with Image.open(gt_path) as image:
            gt = np.array(image)
    else:
        import nibabel as nib
        pred_image, gt_image = nib.load(pred_path), nib.load(gt_path)
        pred, gt = np.asarray(pred_image.dataobj), np.asarray(gt_image.dataobj)
    _validate_masks(pred, gt, 1, pred_path.name)
    if pred_path.name.endswith(".nii.gz"):
        _validate_nifti_pair(pred_path, gt_path, pred_path.name)
    return pred == 1, gt == 1


def rescore(traces_dir: Path, cells_dir: Path, labels_dir: Path, out_dir: Path) -> dict:
    if out_dir.exists():
        raise FileExistsError(f"preserving existing output: {out_dir}")
    gt_files = _label_files(labels_dir)
    report = {"dice_convention": "fmpool.estimators.dice_per_case (no epsilon; empty/empty = 1)",
              "scorer_convention": "train_nnunet_baseline._binary_dice (+1e-6 denominator; empty/empty = 1)",
              "geometry_policy": "NIfTI: SimpleITKIO size/spacing/origin/direction; PNG: array shape",
              "cells": {}}
    trace_paths = sorted(traces_dir.glob("*_fold?_seed??.json"))
    if not trace_paths:
        raise ValueError("no nnU-Net traces to rescore")
    outputs = {}
    for trace_path in trace_paths:
        trace_sha = determinism.sha256_file(trace_path)
        trace = json.loads(trace_path.read_text())
        prediction_dir = cells_dir / trace_path.with_suffix(".cell").name / "holdout_predictions"
        preds = _label_files(prediction_dir)
        ids = _unique_ids(trace["scored_export_ids"], "scored IDs")
        _same_ids(preds, ids, "prediction IDs")
        _same_ids(gt_files, ids, "holdout GT IDs")
        prediction_hashes = _tree_hashes(prediction_dir)
        if prediction_hashes != trace.get("prediction_files"):
            raise ValueError(f"{trace_path.name}: prediction hashes differ from trace")
        identity = trace.get("run_identity_v2", {})
        if identity.get("identity_sha256") != provenance.digest({k: v for k, v in identity.items() if k != "identity_sha256"}):
            raise ValueError(f"{trace_path.name}: invalid run identity digest")
        config = identity["config"]
        for key in ("task", "fold", "seed"):
            if trace.get(key) != config[key]:
                raise ValueError(f"{trace_path.name}: config mismatch: {key}")
        if trace_path.name != f"{config['task']}_fold{config['fold']}_seed{config['seed']}.json":
            raise ValueError(f"{trace_path.name}: canonical trace filename differs from run identity")
        released_ids = [_release_case_id(cid) for cid in ids] if config["task"].startswith("riga_") else ids
        if trace.get("test_ids") != released_ids:
            raise ValueError(f"{trace_path.name}: ordered released case IDs differ from scored export IDs")
        if trace.get("n_test") != len(ids):
            raise ValueError(f"{trace_path.name}: n_test differs from scored export IDs")
        data = identity["data_identity"]
        _same_ids(ids, _unique_ids(data.get("holdout_ids"), "recorded holdout IDs"), "recorded holdout IDs")
        reference_hashes = _tree_hashes(labels_dir)
        expected_references = {name.removeprefix("labelsTs/"): sha for name, sha in data["raw_files"].items()
                               if name.startswith("labelsTs/")}
        if reference_hashes != expected_references:
            raise ValueError(f"{trace_path.name}: reference hashes differ from training data identity")
        previous = np.asarray(trace["per_case_dice"])
        if previous.shape != (len(ids),) or not np.isfinite(previous).all():
            raise ValueError(f"{trace_path.name}: invalid original Dice vector")
        reader = None
        if any(path.name.endswith(".nii.gz") for path in preds.values()):
            import SimpleITK as sitk
            if data.get("plans", {}).get("image_reader_writer") != "SimpleITKIO":
                raise ValueError("recorded SimpleITKIO plans required for NIfTI rescoring")
            reader = {"name": "SimpleITKIO", "SimpleITK_version": sitk.Version_VersionString()}
        dice, empty_both, empty_pred, empty_gt = [], 0, 0, 0
        for cid in ids:
            pred, gt = _read_pair(preds[cid], gt_files[cid])
            empty_both += int(not pred.any() and not gt.any())
            empty_pred += int(not pred.any() and gt.any())
            empty_gt += int(pred.any() and not gt.any())
            dice.append(float(dice_per_case(pred[None], gt[None])[0]))
        if (_tree_hashes(prediction_dir) != prediction_hashes or _tree_hashes(labels_dir) != reference_hashes
                or determinism.sha256_file(trace_path) != trace_sha):
            raise ValueError(f"{trace_path.name}: inputs changed during rescoring")
        diff = np.abs(np.asarray(dice) - previous)
        out = {**trace, "per_case_dice": dice, "mean_dice": float(np.mean(dice)),
               "dice_convention": report["dice_convention"], "rescored_from": trace_path.name,
               "geometry_policy": report["geometry_policy"], "nifti_reader": reader,
               "prediction_files": prediction_hashes, "reference_files": reference_hashes,
               "rescore_inputs": {"trace": str(trace_path.resolve()), "trace_sha256": trace_sha,
                                  "predictions": str(prediction_dir.resolve()), "references": str(labels_dir.resolve())}}
        outputs[trace_path.name] = out
        report["cells"][trace_path.stem] = {"n": len(ids), "empty_pred_and_gt": empty_both, "empty_pred_only": empty_pred,
                                            "empty_gt_only": empty_gt, "max_abs_dice_diff": float(diff.max()), "nifti_reader": reader}
    cells = report["cells"].values()
    report.update(n_cells=len(report["cells"]), total_empty_pred_and_gt=sum(c["empty_pred_and_gt"] for c in cells),
                  total_empty_pred_only=sum(c["empty_pred_only"] for c in cells),
                  total_empty_gt_only=sum(c["empty_gt_only"] for c in cells),
                  max_abs_dice_diff=max(c["max_abs_dice_diff"] for c in cells))
    # Validate the whole batch before publishing any scores.
    out_dir.mkdir(parents=True, exist_ok=False)
    for name, output in outputs.items():
        _write_json(out_dir / name, output)
    _write_json(out_dir / "rescore_report.json", report)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("--traces-dir", "--cells-dir", "--labels-dir", "--out-dir"):
        p.add_argument(name, type=Path, required=True)
    a = p.parse_args(argv)
    r = rescore(a.traces_dir, a.cells_dir, a.labels_dir, a.out_dir)
    print(json.dumps({k: v for k, v in r.items() if k != "cells"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
