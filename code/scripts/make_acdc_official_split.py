"""Build ``data/splits/acdc_official_split.json``: the official ACDC split (Bernard et al., IEEE TMI 2018).

Train = the 100 official training patients (patient001-100, all of them; no validation split: there is no model
selection), test = the 50 official testing patients (patient101-150) with the official reference segmentations
released on the ACDC site. Case IDs are produced by the loader rule of ``fmpool.datasets.acdc``: every ED/ES slice
whose reference contains LV (label 3), ``patientNNN_frameYY_sliceSSS``. The ACDC root must hold the flattened
``Images/patientNNN_frameYY.nii.gz`` + ``Masks/patientNNN_frameYY_gt.nii.gz`` of all 150 patients.

Usage: make_acdc_official_split.py ACDC_ROOT OUT_JSON RETRIEVAL_UTC [PROVENANCE_JSON]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmpool.datasets import acdc  # noqa: E402

SOURCE = {"paper": "Bernard et al., Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation "
                   "and Diagnosis: Is the Problem Solved?, IEEE TMI 37(11) 2018, doi 10.1109/TMI.2018.2837502",
          "site": "https://www.creatis.insa-lyon.fr/Challenge/acdc/databasesTesting.html",
          "download": "https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb "
                      "(public Girder folders training 63721d7073e9f0047faa0525, testing 6372203a73e9f0047faa117e)",
          "test_references": "provided by the site (databasesTesting.html: 'the corresponding manual references ... are also provided')"}


def build(root: Path, retrieval_utc: str, provenance: dict | None = None) -> dict:
    os.environ["FMPOOL_DATA_ROOT"] = str(Path(root).parent)
    ds = acdc.ACDCLVDataset.__new__(acdc.ACDCLVDataset)
    ds.root, ds.images_dir, ds.masks_dir = Path(root), Path(root) / "Images", Path(root) / "Masks"
    public, all_patients = ds._discover_patients()
    test_patients = [p for p in all_patients if int(p[-3:]) in acdc._TEST_PATIENT_RANGE]
    if len(public) != 100 or len(test_patients) != 50:
        raise ValueError(f"need 100 training + 50 testing patients, found {len(public)} + {len(test_patients)}")
    cases, index = ds._enumerate_cases(public + test_patients, official=True)
    train = [c for c in cases if index[c][0] in set(public)]
    test = [c for c in cases if index[c][0] in set(test_patients)]
    split = {"train": train, "val": [], "test": test, "_all_cases": cases, "case_unit": "slice",
             "split_scheme": acdc.OFFICIAL_SCHEME, "test_role": "official_testing_set_with_released_references",
             "n_train_patients": len(public), "n_test_patients": len(test_patients),
             "n_train_cases": len(train), "n_test_cases": len(test),
             "loader_rule": "ED/ES frames of each patient; slice kept iff reference label 3 (LV) has any positive pixel",
             "task": "acdc_official",
             "provenance": {**SOURCE, "retrieval_utc": retrieval_utc, **(provenance or {})}}
    acdc.validate_acdc_official_split(split, index)
    return split


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4) or argv[0] in ("-h", "--help"):
        raise SystemExit(__doc__)
    root, out, retrieval_utc = argv[:3]
    provenance = json.loads(Path(argv[3]).read_text()) if len(argv) > 3 else None
    split = build(Path(root), retrieval_utc, provenance)
    with Path(out).open("x", encoding="utf-8") as fh:
        json.dump(split, fh, indent=2)
    print(out, split["n_train_cases"], split["n_test_cases"], hashlib.sha256(Path(out).read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
