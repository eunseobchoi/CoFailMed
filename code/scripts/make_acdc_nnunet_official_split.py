#!/usr/bin/env python3
"""Frame-level nnU-Net split for the official ACDC 100/50 patient split (case-identical nnU-Net control).

Source: the committed primary official split (``acdc_official_split.json``: train = the 100 official
training patients, test = the 50 official testing patients). Frames come from the listed ACDC image
files (``patientXXX_frameYY.nii.gz``, ED and ES per patient). Inner folds: 5 patient-level folds over the
training patients (numpy default_rng(seed) shuffle, contiguous 20-patient blocks), the same rule that
``prepare_nnunet_data.py`` uses when deriving folds. ``test`` is the fixed official holdout.

Usage: python make_acdc_nnunet_official_split.py SOURCE_SPLIT_JSON IMAGE_LISTING_TXT OUT_JSON [--seed 42]
       [--reference-hashes SHA256SUM_TXT]   # sha256sum output of the official test references, recorded as provenance
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

FRAME_FILE = re.compile(r"^(patient\d{3})_frame(\d{2})\.nii\.gz$")
N_FOLDS = 5


def _patients(ids) -> set[str]:
    patients = {str(cid)[:10] for cid in ids}
    if any(re.fullmatch(r"patient\d{3}", p) is None for p in patients):
        raise ValueError("source split IDs must start with patientXXX")
    return patients


def build(source: dict, listing, seed: int = 42) -> dict:
    train_p, test_p = sorted(_patients(source["train"])), sorted(_patients(source["test"]))
    if set(train_p) & set(test_p) or not train_p or not test_p:
        raise ValueError("source train/test patients must be disjoint and non-empty")
    frames: dict[str, list[str]] = {}
    for name in listing:
        m = FRAME_FILE.match(str(name).strip())
        if m:
            frames.setdefault(m[1], []).append(f"{m[1]}_frame{m[2]}")
    missing = [p for p in train_p + test_p if len(frames.get(p, [])) != 2]
    if missing:
        raise ValueError(f"patients missing ED/ES frames in the listing: {missing[:5]}")
    frames_of = lambda ps: sorted(f for p in ps for f in frames[p])  # noqa: E731
    shuffled = list(train_p)
    np.random.default_rng(seed).shuffle(shuffled)
    n, folds = len(shuffled), {}
    for k in range(N_FOLDS):
        val_p = set(shuffled[(n * k) // N_FOLDS:(n * (k + 1)) // N_FOLDS])
        folds[f"fold_{k}_train"] = frames_of([p for p in train_p if p not in val_p])
        folds[f"fold_{k}_val"] = frames_of(sorted(val_p))
    return {"task": "acdc_lv", "case_unit": "frame_volume", "train": frames_of(train_p), "val": [],
            "test": frames_of(test_p), "folds": folds,
            "provenance": {"scheme": "acdc_official_100_50_patients", "inner_folds": f"{N_FOLDS} patient-level folds over the "
                           f"official training patients, numpy default_rng({seed}) shuffle, contiguous blocks",
                           "test_role": "official ACDC testing patients (101-150) with the official references; "
                                        "every fold/seed model is scored on all of their ED/ES frames"}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source"); ap.add_argument("listing"); ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reference-hashes", type=Path)
    a = ap.parse_args(argv)
    source_path, out = Path(a.source), Path(a.out)
    if out.exists():
        raise FileExistsError(f"preserve existing split: {out}")
    split = build(json.loads(source_path.read_text()), Path(a.listing).read_text().split(), seed=a.seed)
    split["provenance"]["source_split"] = {"path": str(source_path), "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()}
    if a.reference_hashes:
        split["provenance"]["test_reference_sha256"] = dict(
            reversed(line.split()) for line in a.reference_hashes.read_text().splitlines() if line.strip())
    out.write_text(json.dumps(split, indent=2) + "\n")
    print(f"wrote {out}: train {len(split['train'])} test {len(split['test'])} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
