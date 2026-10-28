#!/usr/bin/env python3
"""Strict fold/Magrabia scoring from a completed, hash-pinned v2 training cell."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from fmpool import determinism, provenance
from train_nnunet_baseline import (
    _collect_per_case_dice, _failure, _label_files, _same_ids,
    _tree_hashes, _unique_ids, _upstream_identity, _validate_nifti_pair, _write_json,
)


_FM_LABEL = {
    ("riga_cup", "2d", "nnUNetTrainer_100epochs"): "nnunet_2d_100ep",
    ("riga_cup", "2d", "nnUNetTrainer_1000epochs"): "nnunet_2d_1000ep",
    ("acdc_lv", "3d_fullres", "nnUNetTrainer_1000epochs"): "nnunet_3d_fullres",
    ("acdc_lv", "3d_fullres", "nnUNetTrainer_100epochs"): "nnunet_3d_fullres_100ep",
}


def _release_case_id(case_id):
    return "riga_" + hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:16]


def score_training(evidence_path: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"preserving existing output: {output}")
    record = json.loads(evidence_path.read_text())
    if record.get("schema_version") != "nnunet_cell_v2" or record.get("status") != "trained":
        raise ValueError("completed nnunet_cell_v2 training evidence required")
    identity = record["run_identity_v2"]
    if identity["identity_sha256"] != provenance.digest({k: v for k, v in identity.items() if k != "identity_sha256"}):
        raise ValueError("invalid run identity digest")
    data = record["data"]
    if identity["data_identity"] != data:
        raise ValueError("training data identity mismatch")
    for key in ("task", "dataset_id", "config", "fold", "seed", "base_trainer", "evaluation"):
        if identity["config"][key] != record[key]:
            raise ValueError(f"training config mismatch: {key}")
    if data["upstream"] != _upstream_identity():
        raise ValueError("upstream source changed since training")
    for name, source in record["entrypoint_sources"].items():
        if Path(name).name != name or (Path(__file__).parent / name).read_text() != source:
            raise ValueError("entrypoint source changed since training")
    env = record["environment"]
    raw = Path(env["nnUNet_raw"]) / data["dataset"]
    pre = Path(env["nnUNet_preprocessed"]) / data["dataset"]
    if record["evaluation"] == "holdout":
        holdout_ids = _unique_ids(data.get("holdout_ids"), "holdout IDs")
        metadata = raw / "fmpool_test_ids.json"
        if not metadata.is_file():
            raise ValueError("holdout evaluation requires explicit holdout IDs in fmpool_test_ids.json")
        canonical = _unique_ids(json.loads(metadata.read_text())["test_ids"], "canonical holdout IDs")
        _same_ids(holdout_ids, canonical, "canonical holdout IDs")
        if set(holdout_ids) & set(data["fold"]["train"] + data["fold"]["val"]):
            raise ValueError("holdout overlaps training pool")
    if _tree_hashes(raw) != data["raw_files"] or _tree_hashes(pre) != record["consumed_preprocessed_files"]:
        raise ValueError("consumed raw/preprocessed files changed since training")
    model = Path(record["model"])
    expected_model = Path(env["nnUNet_results"]) / data["dataset"] / f"{record['trainer']}__nnUNetPlans__{record['config']}"
    if model.resolve() != expected_model.resolve():
        raise ValueError("training model path mismatch")
    for name, expected in (("plans.json", data["plans"]), ("dataset.json", data["dataset_json"])):
        if json.loads((model / name).read_text()) != expected:
            raise ValueError(f"model {name} differs from consumed preprocessing")
    fold_dir = model / f"fold_{record['fold']}"
    checkpoint = fold_dir / "checkpoint_final.pth"
    if determinism.sha256_file(checkpoint) != record["checkpoint_sha256"]:
        raise ValueError("training checkpoint hash mismatch")
    _same_ids(_label_files(raw / "labelsTr"), data["fold"]["train"] + data["fold"]["val"], "GT universe")
    prediction_dir = fold_dir / "validation"
    reference_dir = raw / "labelsTr"
    ids, scores = _collect_per_case_dice(prediction_dir, reference_dir, 1,
                                        expected_ids=data["fold"]["val"])
    if record["evaluation"] == "holdout":
        _same_ids(_label_files(raw / "labelsTs"), holdout_ids, "holdout GT IDs")
        prediction_dir = evidence_path.parent / "holdout_predictions"
        prediction_dir.mkdir(exist_ok=False)
        # Explicit final checkpoint: upstream prediction defaults to checkpoint_final,
        # but record the choice instead of relying on a mutable/default recipe.
        command = [sys.executable, "-c",
                   "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()",
                   "-d", data["dataset"], "-i", str(raw / "imagesTs"),
                   "-o", str(prediction_dir), "-c", record["config"], "-f", str(record["fold"]),
                   "-tr", record["trainer"], "-p", "nnUNetPlans", "-chk", "checkpoint_final.pth"]
        subprocess.run(command, env=dict(os.environ, **env, PYTHONDONTWRITEBYTECODE="1"), check=True)
        reference_dir = raw / "labelsTs"
        ids, scores = _collect_per_case_dice(prediction_dir, reference_dir, 1, expected_ids=holdout_ids)
    for cid, prediction in _label_files(prediction_dir).items():
        if prediction.name.endswith(".nii.gz"):
            _validate_nifti_pair(prediction, reference_dir / prediction.name, cid)
    fm = "nnunet"
    released_ids = ids
    if record["evaluation"] == "holdout":
        fm = _FM_LABEL.get((record["task"], record["config"], record["base_trainer"]), fm)
        if record["task"].startswith("riga_"):
            released_ids = [_release_case_id(cid) for cid in ids]
    payload = {"run_identity_v2": identity, "training_evidence_sha256": determinism.sha256_file(evidence_path),
               "task": record["task"], "fm": fm, "fold": record["fold"], "seed": record["seed"],
               "config": record["config"], "evaluation": record["evaluation"], "n_test": len(ids),
               "test_ids": released_ids, "scored_export_ids": ids, "per_case_dice": scores, "mean_dice": float(np.mean(scores)),
               "checkpoint_sha256": record["checkpoint_sha256"],
               "prediction_files": _tree_hashes(prediction_dir),
               "training_elapsed_s": record["training_elapsed_s"]}
    _write_json(output, payload)
    return payload


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-evidence", type=Path, required=True)
    parser.add_argument("--training-evidence-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"preserving existing output: {args.out}")
    try:
        if determinism.sha256_file(args.training_evidence) != args.training_evidence_sha256:
            raise ValueError("training evidence hash mismatch")
        score_training(args.training_evidence, args.out)
    except Exception as exc:
        _failure(args.out, exc)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
