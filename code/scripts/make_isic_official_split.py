"""Build ``data/splits/isic2018_official_split.json``: the official ISIC-2018 Task 1 training / test split.

Train = the official Task 1 training set (2594 images with public ground truth), test = the official Task 1 TEST set
(1000 images; ground truth public since the challenge closed). Case IDs are the paired (image, mask) ISIC stems found
in the four official folders under ISIC_ROOT (``ISIC2018_Task1-2_{Training,Test}_Input``,
``ISIC2018_Task1_{Training,Test}_GroundTruth``), sorted by ID; the loader rule (mask > 127) is unchanged.

Usage: make_isic_official_split.py ISIC_ROOT OUT_JSON RETRIEVAL_UTC [PROVENANCE_JSON]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmpool.datasets import isic2018  # noqa: E402

SOURCE = {"paper": "Codella et al., Skin Lesion Analysis Toward Melanoma Detection 2018 (arXiv 1902.03368); Tschandl et al., "
                   "The HAM10000 dataset, Sci Data 2018, doi 10.1038/sdata.2018.161",
          "site": "https://challenge.isic-archive.com/data/#2018",
          "archives": {role: f"https://isic-archive.s3.amazonaws.com/challenges/2018/{name}.zip" for role, name in (
              ("train_input", "ISIC2018_Task1-2_Training_Input"), ("train_gt", "ISIC2018_Task1_Training_GroundTruth"),
              ("test_input", "ISIC2018_Task1-2_Test_Input"), ("test_gt", "ISIC2018_Task1_Test_GroundTruth"))},
          "licence": "CC0 1.0 (LICENSE.txt / ATTRIBUTION.txt inside the ground-truth archives)"}


def cases(root: Path, images_sub: str, masks_sub: str) -> list[str]:
    ds = isic2018.ISIC2018Dataset.__new__(isic2018.ISIC2018Dataset)
    ds.images_dir, ds.masks_dir = root / images_sub, root / masks_sub
    return ds._discover_cases()


def build(root: Path, provenance: dict, retrieval_utc: str) -> dict:
    train = cases(root, *isic2018.OFFICIAL_DIRS["train"])
    test = cases(root, *isic2018.OFFICIAL_DIRS["test"])
    split = {"train": train, "val": [], "test": test, "split_scheme": isic2018.OFFICIAL_SCHEME,
            "test_role": "official_task1_test_set_with_released_ground_truth",
            "n_train": len(train), "n_test": len(test), "folders": isic2018.OFFICIAL_DIRS, "task": "isic2018_official",
            "provenance": {**SOURCE, "retrieval_utc": retrieval_utc, **provenance}}
    isic2018.validate_isic_official_split(split)
    return split


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4) or argv[0] in ("-h", "--help"):
        raise SystemExit(__doc__)
    root, out, retrieval_utc = argv[:3]
    provenance = json.loads(Path(argv[3]).read_text()) if len(argv) > 3 else {}
    split = build(Path(root), provenance, retrieval_utc)
    with Path(out).open("x", encoding="utf-8") as fh:
        json.dump(split, fh, indent=2)
    print(out, split["n_train"], split["n_test"], hashlib.sha256(Path(out).read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
