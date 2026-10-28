"""Write the nnU-Net input_manifest_v2 for one exported task.

Writes the manifest and <manifest>.split.json (the committed split in nnU-Net export form).
Each option defaults to the environment variable named in its help text.

example:
  python3 code/scripts/make_nnunet_input_manifest.py --task riga_cup \\
    --raw-export <nnUNet_raw>/Dataset501_RIGACup --receipt <preprocessing receipt> \\
    --canonical-split code/data/splits/riga_cup_split.json --input-manifest <out.json>
"""
import argparse
import json
import os
from pathlib import Path

OPTIONS = (
    ("--task", "TASK", "task name, e.g. riga_cup or acdc_lv"),
    ("--raw-export", "RAW_EXPORT", "nnU-Net raw dataset directory"),
    ("--receipt", "RECEIPT", "preprocessing receipt JSON"),
    ("--canonical-split", "CANONICAL_SPLIT", "committed split under code/data/splits"),
    ("--input-manifest", "INPUT_MANIFEST", "output manifest path (must not exist)"),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag, env, text in OPTIONS:
        parser.add_argument(flag, default=os.environ.get(env), required=env not in os.environ,
                            help=f"{text} (default: ${env})")
    args = parser.parse_args(argv)
    # Imported after parsing so that --help works without the training environment.
    from fmpool import provenance, determinism
    from train_nnunet_baseline import _tree_hashes, _write_json

    task = args.task
    raw = Path(args.raw_export).resolve()
    receipt = Path(args.receipt).resolve()
    source_path = Path(args.canonical_split).resolve()
    output = Path(args.input_manifest).resolve()
    split_path = output.with_suffix(".split.json")
    if output.exists() or split_path.exists():
        raise FileExistsError("preserve existing manifest/split")
    source = json.loads(source_path.read_text())
    if task == "acdc_lv" and source.get("case_unit") == "frame_volume":
        # Frame-level nnU-Net split (official ACDC 100/50 rerun): built by make_acdc_nnunet_official_split.py from the
        # committed primary official split; validated under the nnunet_export contract below.
        split = source
    elif task == "acdc_lv":
        provenance.validate_source_split(task, source)  # Full primary validation stays on.
        def frames(ids):
            return list(dict.fromkeys(cid.split("_slice")[0] for cid in ids))
        split = {role: frames(source[role]) for role in ("train", "val", "test")}
        split.update(case_unit="frame_volume",
                     folds={name: frames(ids) for name, ids in source["folds"].items()})
    else:
        provenance.validate_source_split(task, source)  # Full primary validation stays on.
        split = source  # Keep original RIGA IDs/canonical image mapping.
    provenance.validate_source_split(task, split, split_contract="nnunet_export")
    record = json.loads(receipt.read_text())
    raw_hashes = _tree_hashes(raw)
    if record["dataset"] != raw.name or record["raw_files"] != raw_hashes:
        raise ValueError("receipt differs from the raw export")
    _write_json(split_path, split)
    raw_files = {str(raw / name): sha for name, sha in raw_hashes.items()}
    raw_files[str(source_path)] = determinism.sha256_file(source_path)
    _write_json(output, {
        "schema_version": "input_manifest_v2", "task": task,
        "raw_files": raw_files,
        "source_split_manifest": str(source_path),
        "split_manifest": str(split_path),
        "split_manifest_sha256": determinism.sha256_file(split_path),
        "nnunet_preprocessing_sha256": determinism.sha256_file(receipt),
    })
    print("manifest written", output)


if __name__ == "__main__":
    main()
