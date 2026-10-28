#!/usr/bin/env python3
"""Write the ``input_manifest_v2`` JSON that the trainers take via ``--input-manifest``.

The manifest pins every file under the task's dataset root (absolute path -> sha256), the committed
split file, and the encoder: the declared weight source (repository + revision) and the sha256 of the
loaded FP32 encoder state, which the trainers re-check before training.
Use --multiscale for train_head_unet_skip.py; the default pins the single-scale wrapper.

usage: PYTHONPATH=code/src python3 code/scripts/make_input_manifest.py --task riga_cup \
           --fm dinov2_vitb14 --repository facebookresearch/dinov2 --revision <commit> \
           --out inputs/riga_cup/dinov2_vitb14.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fmpool import datasets, determinism, encoders, provenance


def build_manifest(task: str, fm: str, repository: str, revision: str, *, multiscale: bool = False) -> dict:
    if repository == "facebookresearch/dinov2":
        os.environ["FMPOOL_DINOV2_REVISION"] = revision  # same hub revision the trainers will load
    if multiscale:
        from fmpool.multiscale import build_multiscale_encoder

        encoder, _ = build_multiscale_encoder(fm)
    else:
        encoder, _ = encoders.build_encoder(fm)
    state_sha256 = provenance.state_sha256(encoder.float())  # trainers hash the FP32 encoder
    roots = {Path(datasets.build_dataset(task, role).root).resolve() for role in ("train", "test")}
    files = sorted(path for root in roots for path in root.rglob("*"))
    if any(path.is_symlink() for path in files):
        raise ValueError("symlinks inside the dataset root are not supported; point FMPOOL_DATA_ROOT at real files")
    split = datasets.splits_path(task).resolve()
    return {"schema_version": "input_manifest_v2", "task": task,
            "raw_files": {str(path.resolve()): determinism.sha256_file(path) for path in files if path.is_file()},
            "split_manifest": str(split), "split_manifest_sha256": determinism.sha256_file(split),
            "encoder_source": {"fm": fm, "repository": repository, "revision": revision,
                               "state_sha256": state_sha256}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--fm", required=True)
    parser.add_argument("--repository", required=True, help="weight source, e.g. pytorch/vision or an HF repo id")
    parser.add_argument("--revision", required=True, help="library version, git commit or HF snapshot of the weights")
    parser.add_argument("--multiscale", action="store_true",
                        help="pin the multiscale wrapper used by train_head_unet_skip.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_manifest(args.task, args.fm, args.repository, args.revision,
                              multiscale=args.multiscale)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:  # never overwrite a pinned manifest
        json.dump(manifest, handle, indent=1, sort_keys=True)
    provenance.read_input_manifest(args.out, args.task)  # same validation the trainers run
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
